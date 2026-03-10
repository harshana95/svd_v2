# from PyQt5.QtCore import QLibraryInfo
# import os
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

import os
import argparse
from PIL import Image
import torch
import einops
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchmetrics.image.arniqa import ARNIQA
import pyiqa
import numpy as np
from skimage.exposure import match_histograms

def load_images_from_folder(folder, image_size):
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    images = []
    for fname in image_files:
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        img = img.resize((image_size, image_size))
        images.append(img)
    return images, image_files

def match_and_resize_images(gt_images, pred_images):
    matched_gt_images = []
    matched_pred_images = []
    for gt_img, pred_img in zip(gt_images, pred_images):
        gt_w, gt_h = gt_img.size
        pred_w, pred_h = pred_img.size
        if (gt_w, gt_h) != (pred_w, pred_h):
            new_w = max(gt_w, pred_w)
            new_h = max(gt_h, pred_h)
            print(new_h, new_w, gt_h, gt_w)
            gt_img = gt_img.resize((new_w, new_h))
            pred_img = pred_img.resize((new_w, new_h))
        matched_gt_images.append(gt_img)
        matched_pred_images.append(pred_img)
    return matched_gt_images, matched_pred_images

def match_colors_by_histogram(gt_image, pred_image):
    """
    Matches the colors of the predicted image to the ground truth image using histogram matching.
    """
    gt_array = np.array(gt_image)
    pred_array = np.array(pred_image)
    if len(pred_array.shape) == 2:
        gt_array = einops.repeat(gt_array, 'h w -> h w c', c=3)
        pred_array = einops.repeat(pred_array, 'h w -> h w c', c=3)

    # Check if images have the same shape
    if gt_array.shape != pred_array.shape:
        raise ValueError("Ground truth and predicted images must have the same dimensions for histogram matching.")
    
    matched_array = np.zeros_like(pred_array)
    for i in range(3):  # Process each channel (R, G, B)
        matched_array[:, :, i] = match_histograms(pred_array[:, :, i], gt_array[:, :, i])
    
    return Image.fromarray(matched_array.astype(np.uint8)), Image.fromarray(gt_array.astype(np.uint8))

def save_image_pairs(gt_images, pred_images, pred_folder):
    pairs_folder = os.path.join(pred_folder, "pairs")
    os.makedirs(pairs_folder, exist_ok=True)
    print(f"Saving pairs in {pairs_folder}")

    for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gt_img.transpose([1,2,0]))
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(pred_img.transpose([1,2,0]))
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(pairs_folder, f"pair_{i + 1}.png")
        plt.savefig(save_path)
        plt.close(fig)

def move_and_compute(metric, pred, gt, pixels=1):
    results = []
    metric.reset()
    # 1 c h w
    for i in range(pixels):
        for j in range(pixels):
            _pred = torch.roll(pred, shifts=(i,j), dims=(2, 3))
            metric.update(gt[:, :, pixels:-pixels, pixels:-pixels], _pred[:, :, pixels:-pixels, pixels:-pixels])
            metric_value = metric.compute().item()
            results.append(metric_value)
            metric.reset()
    return results

def get_random_crop(images, size):
    _, _, h, w = images[0].shape
    ih = np.random.randint(0, h-size[0])
    iw = np.random.randint(0, w-size[1])
    return [image[:, :, ih:ih+size[0], iw:iw+size[1]] for image in images]


@torch.no_grad()
def main(gt_folder, pred_folder, image_size, device='cuda'):
    tmp = pred_folder
    while os.path.basename(os.path.dirname(tmp)) != "Results":
        tmp = os.path.dirname(tmp)
    name = os.path.basename(tmp)
    print(name)

    # Load images
    gt_images, gt_names = load_images_from_folder(gt_folder, image_size)
    pred_images, pred_names = load_images_from_folder(pred_folder, image_size)
    
    # change channels if necessary
    if args.channel in ['r', 'g', 'b']:
        idx = 'rgb'.index(args.channel)
        gt_idx = 'rgb'.index(args.channel_gt)
        gt_images = [img.split()[gt_idx] for img in gt_images]
        pred_images = [img.split()[idx] for img in pred_images]

    gt_images, pred_images = match_and_resize_images(gt_images, pred_images)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Perform color matching on predicted images
    matched_pred_images = []
    i=0
    for gt_img, pred_img in zip(gt_images, pred_images):
        matched_img, new_gt = match_colors_by_histogram(gt_img, pred_img)
        matched_pred_images.append(matched_img)
        gt_images[i] = new_gt
        i+=1

    gt_tensors = torch.stack([transform(img) for img in gt_images]).to(device)
    matched_pred_tensors = torch.stack([transform(img) for img in matched_pred_images]).to(device)

    save_image_pairs(gt_tensors.cpu().numpy(), matched_pred_tensors.cpu().numpy(), pred_folder)
    os.makedirs(os.path.join(pred_folder, 'out'), exist_ok=True)
    
    for i in range(len(matched_pred_tensors)):
        plt.imsave(os.path.join(pred_folder, 'out', f"{name}_{i + 1}.jpg"), matched_pred_tensors[i].cpu().numpy().transpose([1,2,0]))

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # data_range specifies the range of pixel values (e.g., 1.0 for [0, 1] or 255.0 for [0, 255])
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # net_type can be 'alex', 'vgg', or 'squeeze'
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    # if normalize is true values should be [0,1] else [0,255]
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    dists = pyiqa.create_metric('dists', device=device)
    niqe = pyiqa.create_metric('niqe', device=device)
    musiq = pyiqa.create_metric('musiq', device=device)
    maniqa = pyiqa.create_metric('maniqa', device=device)
    clipiqa = pyiqa.create_metric('clipiqa', device=device)
    n_rand_crop_for_fid = 64
    values = {"lpips": [], "ssim":[], "fid":[], 'psnr':[],
                  'dists':[], 'niqe':[], 'musiq':[], 'maniqa':[], 'clipiqa':[]}
    for gt_img, pred_img in zip(gt_images, matched_pred_images):
        
        # Convert images to tensors
        gt = transform(gt_img).unsqueeze(0).to(device)
        pred = transform(pred_img).unsqueeze(0).to(device)
        
        values['psnr'].append(max(move_and_compute(psnr, pred, gt)))
        values['ssim'].append(max(move_and_compute(ssim, pred, gt)))

        
        # Compute the LPIPS score
        # with shape [N, 3, H, W] and normalized to [-1, 1]
        values['lpips'].append(min(move_and_compute(lpips, pred*2-1, gt*2-1)))
        values['dists'].append(dists(pred, gt))
        
        # Compute the FID score
        # Update the metric with real and fake images
        gt_batch = []
        pred_batch = []
        for i in range(n_rand_crop_for_fid):
            gt_crop, pred_crop = get_random_crop([gt,pred], (299,299))
            gt_batch.append(gt_crop)
            pred_batch.append(pred_crop)
            if len(gt_batch) == 16 or i==n_rand_crop_for_fid-1:
                fid.update(torch.concat(gt_batch, dim=0) , real=True)
                fid.update(torch.concat(pred_batch, dim=0), real=False)
                gt_batch = []
                pred_batch = []
        
        # no-reference metrics
        values['niqe'].append(niqe(pred))
        values['musiq'].append(musiq(pred))
        values['maniqa'].append(maniqa(pred))
        values['clipiqa'].append(clipiqa(pred))

    
    fid_score = fid.compute()
    values['fid'].append(fid_score.item())
    fid.reset()

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # FID (expects images in [0, 255] uint8, shape [N, 3, H, W])
    fid = FrechetInceptionDistance(feature=64).to(device) #, feature_extractor_weights_path="/depot/chan129/users/harshana/inception_v3_google-1a9a5a14.pth").to(device)
    gt_uint8 = (gt_tensors * 255).byte()
    pred_uint8 = (matched_pred_tensors * 255).byte()
    fid.update(gt_uint8, real=True)
    fid.update(pred_uint8, real=False)
    fid_value = fid.compute().item()

    keys = ['psnr', 'ssim', 'lpips', 'dists', 'fid', 'niqe', 'musiq', 'maniqa', 'clipiqa']
    to_print = f"{pred_folder}"
    for key in keys:
        value = torch.mean(torch.tensor(values[key])).cpu().item()
        to_print += f" & {value:.4f}"
    to_print += " \\\\"
    print(to_print)

    # Append results to results.txt
    results_file = os.path.join(gt_folder, "results.txt")
    with open(results_file, "a") as f:
        f.write(f"{to_print}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PSNR, LPIPS, FID between GT and predicted images.")
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to ground truth image folder')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to predicted image folder')
    parser.add_argument('--image_size', type=int, required=True, help='resize to image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--channel', type=str, default='none', help='channel', choices=['none', 'r', 'g', 'b'])
    parser.add_argument('--channel_gt', type=str, default='none', help='channel', choices=['none', 'r', 'g', 'b'])
    args = parser.parse_args()
    main(args.gt_folder, args.pred_folder, args.image_size, args.device)
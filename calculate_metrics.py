import os
import argparse
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity
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
    
    # Check if images have the same shape
    if gt_array.shape != pred_array.shape:
        raise ValueError("Ground truth and predicted images must have the same dimensions for histogram matching.")
    
    matched_array = np.zeros_like(pred_array)
    for i in range(3):  # Process each channel (R, G, B)
        matched_array[:, :, i] = match_histograms(pred_array[:, :, i], gt_array[:, :, i])
    
    return Image.fromarray(matched_array.astype(np.uint8))

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
        
@torch.no_grad()
def main(gt_folder, pred_folder, image_size, device='cuda'):
    # Load images
    gt_images, gt_names = load_images_from_folder(gt_folder, image_size)
    pred_images, pred_names = load_images_from_folder(pred_folder, image_size)

    gt_images, pred_images = match_and_resize_images(gt_images, pred_images)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Perform color matching on predicted images
    matched_pred_images = [match_colors_by_histogram(gt_img, pred_img) for gt_img, pred_img in zip(gt_images, pred_images)]

    gt_tensors = torch.stack([transform(img) for img in gt_images]).to(device)
    matched_pred_tensors = torch.stack([transform(img) for img in matched_pred_images]).to(device)

    save_image_pairs(gt_tensors.cpu().numpy(), matched_pred_tensors.cpu().numpy(), pred_folder)
    os.makedirs(os.path.join(pred_folder, 'out'), exist_ok=True)
    tmp = pred_folder
    while os.path.basename(os.path.dirname(tmp)) != "Results":
        tmp = os.path.dirname(tmp)
    name = os.path.basename(tmp)
    print(name)
    for i in range(len(matched_pred_tensors)):
        plt.imsave(os.path.join(pred_folder, 'out', f"{name}_{i + 1}.png"), matched_pred_tensors[i].cpu().numpy().transpose([1,2,0]))

    # PSNR
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr_values = []

    # LPIPS
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    lpips_values = []

    for gt_img, pred_img in zip(gt_images, matched_pred_images):
        
        # Convert images to tensors
        gt_tensor = transform(gt_img).unsqueeze(0).to(device)
        pred_tensor = transform(pred_img).unsqueeze(0).to(device)
        
        # Calculate PSNR for the current pair
        psnr_value = psnr(pred_tensor, gt_tensor).item()
        psnr_values.append(psnr_value)

        # Calculate LPIPS for the current pair
        lpips_value = lpips(pred_tensor, gt_tensor).item()
        lpips_values.append(lpips_value)

    avg_psnr_value = sum(psnr_values) / len(psnr_values)
    avg_lpips_value = sum(lpips_values) / len(lpips_values)

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # FID (expects images in [0, 255] uint8, shape [N, 3, H, W])
    fid = FrechetInceptionDistance(feature=64).to(device) #, feature_extractor_weights_path="/depot/chan129/users/harshana/inception_v3_google-1a9a5a14.pth").to(device)
    gt_uint8 = (gt_tensors * 255).byte()
    pred_uint8 = (matched_pred_tensors * 255).byte()
    fid.update(gt_uint8, real=True)
    fid.update(pred_uint8, real=False)
    fid_value = fid.compute().item()

    print(f" {'='*10} {pred_folder}")
    print(f"PSNR: {avg_psnr_value:.4f}")
    print(f"LPIPS: {avg_lpips_value:.4f}")
    print(f"FID: {fid_value:.4f}")
    print(f"{avg_psnr_value:.4f}\t{avg_lpips_value:.4f}\t{fid_value:.4f}")

    # Append results to results.txt
    results_file = os.path.join(gt_folder, "results.txt")
    with open(results_file, "a") as f:
        f.write(f" {'='*10} {pred_folder}\n")
        f.write(f"PSNR: {avg_psnr_value:.4f}, LPIPS: {avg_lpips_value:.4f}, FID: {fid_value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PSNR, LPIPS, FID between GT and predicted images.")
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to ground truth image folder')
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to predicted image folder')
    parser.add_argument('--image_size', type=int, required=True, help='resize to image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    main(args.gt_folder, args.pred_folder, args.image_size, args.device)
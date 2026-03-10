
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
from PIL import Image
from skimage.exposure import match_histograms

def load_images_from_folder(folder, image_size):
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    images = []
    for fname in image_files:
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        img = img.resize((image_size, image_size))
        images.append(img)
    print(f"Loaded {len(images)} images from {folder}")
    return images, image_files


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


def plot_chromaticity_diagram(gt_images, pred_images, method_name, folder_path):
    """
    Plots the chromaticity diagram for GT and Predicted images to visualize color distribution.
    """
    import matplotlib.colors as mcolors
    
    def get_chromaticity(img_list):
        r_coords = []
        g_coords = []
        for img in img_list:
            # Flatten and avoid division by zero
            img_flat = einops.rearrange(np.array(img).astype(np.float32)/255.0, 'h w c -> (h w) c')
            sum_rgb = np.sum(img_flat, axis=1, keepdims=True)
            sum_rgb[sum_rgb == 0] = 1e-6
            # breakpoint()
            chroma = img_flat / sum_rgb
            r_coords.append(chroma[:, 0])
            g_coords.append(chroma[:, 1])
        return np.concatenate(r_coords), np.concatenate(g_coords)

    gt_r, gt_g = get_chromaticity(gt_images)
    pred_r, pred_g = get_chromaticity(pred_images)

    # Subsample for better visualization (too many points can be cluttered)
    subsample_factor = max(1, len(gt_r) // 10000)
    gt_r_sub = gt_r[::subsample_factor]
    gt_g_sub = gt_g[::subsample_factor]
    pred_r_sub = pred_r[::subsample_factor]
    pred_g_sub = pred_g[::subsample_factor]

    plt.figure(figsize=(8, 8))
    
    # Use scatter plots with transparency for chromaticity visualization
    plt.scatter(gt_r_sub, gt_g_sub, c='blue', alpha=0.3, s=1, label='Ground Truth', marker='.')
    plt.scatter(pred_r_sub, pred_g_sub, c='red', alpha=0.3, s=1, label='Prediction', marker='.')
    
    # Spectral locus approximation (simplified)
    plt.plot([0, 0, 1, 0], [0, 1, 0, 0], 'k--', label='RGB Gamut Boundary', linewidth=1.5)
    
    plt.xlabel('r = R/(R+G+B)')
    plt.ylabel('g = G/(R+G+B)')
    plt.title(f'Chromaticity Diagram: {method_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    save_path = os.path.join(folder_path, f"{method_name}_chromaticity.png")
    plt.savefig(save_path)
    plt.close()

def calculate_mean_color_difference(gt_images, pred_images):
    """
    Calculate the mean color difference (per image) in LAB color space (Delta E 2000).
    Returns a list of mean Delta E values (one per image).
    """
    import cv2
    import numpy as np
    from skimage.color import rgb2lab, deltaE_ciede2000

    delta_e_means = []
    for gt_img, pred_img in zip(gt_images, pred_images):
        # Ensure both images are numpy arrays and uint8
        gt_np = np.array(gt_img).astype(np.uint8)
        pred_np = np.array(pred_img).astype(np.uint8)
        # Resize predicted to match GT if necessary
        if gt_np.shape != pred_np.shape:
            pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Convert RGB to LAB
        lab_gt  = rgb2lab(gt_np / 255.0)
        lab_pred = rgb2lab(pred_np / 255.0)
        # Reshape to (num_pixels, 3)
        lab_gt_flat = lab_gt.reshape(-1, 3)
        lab_pred_flat = lab_pred.reshape(-1, 3)
        # Compute Delta E per pixel
        delta_e = deltaE_ciede2000(lab_gt_flat, lab_pred_flat)
        delta_e_means.append(np.mean(delta_e))
    return delta_e_means
def calculate_cni(gt_images, pred_images):
    """
    Calculate Color Naturalness Index (CNI) per image.
    CNI measures how natural the colors in the prediction are, based on distance from typical natural color statistics in ab-plane of Lab.
    Returns a list of CNI values (one per image), higher is more natural.
    Adapted from: Yendrikhovskij, S. N. (1998). "Color reproduction and the naturalness constraint."
    """
    import numpy as np
    from skimage.color import rgb2lab

    # Reference natural a*, b* distribution for natural images (typical mean and std)
    # These are standard values from literature, can be improved for your datasets
    NATURAL_MEAN_AB = np.array([0, 0])
    NATURAL_STD_AB = np.array([25, 25])  # Change as needed

    cni_scores = []
    for img in pred_images:
        arr_rgb = np.array(img).astype(np.float32) / 255.0
        arr_lab = rgb2lab(arr_rgb)
        ab = arr_lab[..., 1:3].reshape(-1, 2)
        mean_ab = np.mean(ab, axis=0)
        dist = np.linalg.norm((mean_ab - NATURAL_MEAN_AB) / NATURAL_STD_AB)
        # CNI is designed so that lower distance from natural=more natural (invert/saturate to get [0..1])
        cni = np.exp(-dist)
        cni_scores.append(cni)
    return cni_scores

def calculate_cci(gt_images, pred_images):
    """
    Calculate Colourfulness Index (CCI) per image as proposed by Hasler & Suesstrunk (2003).
    Returns a list of CCI values (one per image), higher is more colorful.
    """
    import numpy as np

    cci_scores = []
    for img in pred_images:
        arr = np.array(img).astype(np.float32)
        R = arr[..., 0]
        G = arr[..., 1]
        B = arr[..., 2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        # Colourfulness index formula from Hasler & Suesstrunk (2003)
        cci = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        cci_scores.append(cci)
    return cci_scores

def plot_color_difference_bar_chart(delta_e_means, folder_path):
    """
    Plot a bar chart of mean color difference (Delta E 2000) per image.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    values = []
    keys = []
    for key in delta_e_means:
        values.append(delta_e_means[key])
        keys.append(key)
    values = np.array(values)
    
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)
    plt.figure(figsize=(10, 5))
    plt.bar(keys, mean, yerr=std, color='mediumseagreen')
    plt.xlabel('Method')
    plt.ylabel('Mean Color Difference (ΔE 2000)')
    # plt.title(f'Color Difference')
    plt.tight_layout()
    save_path = os.path.join(folder_path, f"color_difference_bar.png")
    plt.savefig(save_path)
    plt.close()

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
    matched_pred_images = []
    for gt_img, pred_img in zip(gt_images, pred_images):
        matched_img, new_gt = match_colors_by_histogram(gt_img, pred_img)
        matched_pred_images.append(matched_img)
    
    plot_chromaticity_diagram(gt_images, matched_pred_images, name, os.path.dirname(gt_folder[:-1]))
    
    delta_e_means = calculate_mean_color_difference(gt_images, matched_pred_images)
    return name, delta_e_means

if __name__ == "__main__":
    gt_folder = "/depot/chan129/users/harshana/OpticsExpress_2026/Results/GT/"
    pred_folders = [
        "/depot/chan129/users/harshana/OpticsExpress_2026/Results/Ours/",
        "/depot/chan129/users/harshana/OpticsExpress_2026/Results/Pinilla/",
        "/depot/chan129/users/harshana/OpticsExpress_2026/Results/Tseng/",
        "/depot/chan129/users/harshana/OpticsExpress_2026/Results/NAFNet/",
        "/depot/chan129/users/harshana/OpticsExpress_2026/Results/ResShift/",
        "/depot/chan129/users/harshana/OpticsExpress_2026/Results/DiffBIR/",
        "/depot/chan129/users/harshana/OpticsExpress_2026/Results/PAN-Crafter/",
    ]
    e_delta = {}
    for pred_folder in pred_folders:
        name, ed = main(gt_folder, pred_folder, 512)
        e_delta[name] = ed
    
    plot_color_difference_bar_chart(e_delta, os.path.dirname(gt_folder[:-1]))
import math
import cv2
import random
import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
from scipy.interpolate import Rbf
from torchvision import transforms as T
import torch.nn as nn
import json

class CMOS_QIS_test_dataset_predefined(torch.utils.data.Dataset):
    """
    Dataset that returns: CMOS simulated image, QIS simulated image, and ground truth RGB.
    Simulates both CMOS and QIS sensors from RGB images.
    """
    def __init__(self, opt):
        self.folder = opt['folder']
        self.cmos_folder = opt['cmos_folder']
        self.cmos_path = sorted(glob.glob(os.path.join(self.cmos_folder, '*.png')))
        self.qis_path = sorted(glob.glob(os.path.join(self.folder, 'qis', '*.png')))
        self.gt_path = sorted(glob.glob(os.path.join(self.folder, 'gt', '*.png')))
        self.metadata_path = sorted(glob.glob(os.path.join(self.folder, 'metadata', '*.json')))
        
        self.gt_key = opt['gt_key']
        self.lq_key = opt['lq_key']
    def __len__(self):
        return len(self.cmos_path)
    
    def read_json(self, path):
        # 2. Load JSON
        json_path = os.path.join(path)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def __getitem__(self, idx):
        cmos_img = cv2.imread(self.cmos_path[idx])
        qis_img  = cv2.imread(self.qis_path[idx], cv2.IMREAD_UNCHANGED)
        gt_img   = cv2.imread(self.gt_path[idx])

        h, w = gt_img.shape[:2]
        qis_img = qis_img[:h, :w]

        # --- upscale ×2 ---
        scale = 2
        new_size = (w * scale, h * scale)

        # Bicubic for all
        qis_img  = cv2.resize(qis_img,  new_size, interpolation=cv2.INTER_NEAREST)
        cmos_img = cv2.resize(cmos_img, new_size, interpolation=cv2.INTER_CUBIC)
        gt_img   = cv2.resize(gt_img,   new_size, interpolation=cv2.INTER_CUBIC)

        # ---- restore original 4-bit discrete levels in 0-255 space ----
        # assume uniform quantization (most common case)

        levels = np.linspace(0, 255, 16)   # original 16 photon levels

        qis_img = qis_img.astype(np.float32)

        # project each pixel to nearest valid level
        qis_img = levels[np.argmin(np.abs(qis_img[..., None] - levels), axis=-1)]

        qis_img = qis_img.astype(np.uint8)
        
        gt_gray_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        metadata = self.read_json(self.metadata_path[idx])

        cmos_img = cv2.cvtColor(cmos_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        
        qis_img = qis_img[:,:,0:1]
        gt_gray_img = gt_gray_img[:,:,None]

        cmos_img = torch.from_numpy(cmos_img.astype(np.float32)/255.).permute(2, 0, 1)
        qis_img = torch.from_numpy(qis_img.astype(np.float32)/255.).permute(2, 0, 1)
        gt_img = torch.from_numpy(gt_img.astype(np.float32)/255.).permute(2, 0, 1)
        gt_gray_img = torch.from_numpy(gt_gray_img.astype(np.float32)/255.).permute(2, 0, 1)

        # prompt = metadata['struct_color_prompt_gt']
        prompt = ''
        gain = torch.from_numpy(np.array([metadata['iso_gain']]))
        # print('gain: ', gain, type(gain))
        lux = metadata['lux']

        # Normalize to [-1, 1]
        data = {
            'blur_1': cmos_img * 2 - 1,  # 0,1 to -1,1
            'blur_2': qis_img * 2 - 1,        # 0,1 to -1,1
            'gt_1': gt_img * 2 - 1,       # 0,1 to -1,1
            'gt_2': gt_gray_img * 2 - 1, # 0,1 to -1,1
            'qis_iso_gain': gain/90,
            'struct_color_desc': prompt, 
            'lux': lux
        }
        
        return data

class CMOS_QIS_dataset_predefined(torch.utils.data.Dataset):
    """
    Dataset that returns: CMOS simulated image, QIS simulated image, and ground truth RGB.
    Simulates both CMOS and QIS sensors from RGB images.
    """
    def __init__(self, opt):
        self.folder = opt['folder']
        self.cmos_path = sorted(glob.glob(os.path.join(self.folder, 'cmos', '*.png')))
        self.qis_path = sorted(glob.glob(os.path.join(self.folder, 'qis', '*.png')))
        # self.qis_denoised_path = sorted(glob.glob(os.path.join(self.folder, 'qisden', '*.png')))
        # self.x_anchor_path = sorted(glob.glob(os.path.join(self.folder, 'anchor', '*.png')))
        self.gt_path = sorted(glob.glob(os.path.join(self.folder, 'gt', '*.png')))
        self.metadata_path = sorted(glob.glob(os.path.join(self.folder, 'metadata', '*.json')))

        self.val_lux = opt['val_lux']
        self.mode = opt['mode']
        self.gt_key = opt['gt_key']
        self.lq_key = opt['lq_key']
        
        if self.val_lux:
            self.path_modification()
        
    def __len__(self):
        return len(self.cmos_path)
    
    def path_modification(self):
        if self.mode == 'val':
            self.cmos_path = sorted(glob.glob(os.path.join(self.folder, f'{self.val_lux}_lux', 'cmos', "any_exp", '*.png')))
            self.qis_path = sorted(glob.glob(os.path.join(self.folder, f'{self.val_lux}_lux', 'qis', '*.png')))
            # self.qis_denoised_path = sorted(glob.glob(os.path.join(self.folder, f'{self.val_lux}_lux', 'qisden', '*.png')))
            # self.x_anchor_path = sorted(glob.glob(os.path.join(self.folder, f'{self.val_lux}_lux', 'anchor', '*.png')))
            self.gt_path = sorted(glob.glob(os.path.join(self.folder, f'{self.val_lux}_lux', 'gt', '*.png')))
            self.metadata_path = sorted(glob.glob(os.path.join(self.folder, f'{self.val_lux}_lux', 'metadata', '*.json')))
        else:
            pass
    
    def read_json(self, path):
        # 2. Load JSON
        json_path = os.path.join(path)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def __getitem__(self, idx):
        cmos_img = cv2.imread(self.cmos_path[idx])
        qis_img = cv2.imread(self.qis_path[idx], cv2.IMREAD_UNCHANGED)
        # qisden_img = cv2.imread(self.qis_denoised_path[idx], cv2.IMREAD_UNCHANGED)
        # x_anchor_img = cv2.imread(self.x_anchor_path[idx])
        gt_img = cv2.imread(self.gt_path[idx])
        gt_gray_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        metadata = self.read_json(self.metadata_path[idx])

        cmos_img = cv2.cvtColor(cmos_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        # x_anchor_img = cv2.cvtColor(x_anchor_img, cv2.COLOR_BGR2RGB)
        
        qis_img = qis_img[:,:,0:1]
        # qisden_img = qisden_img[:,:,0:1]
        gt_gray_img = gt_gray_img[:,:,None]

        cmos_img = torch.from_numpy(cmos_img.astype(np.float32)/255.).permute(2, 0, 1)
        qis_img = torch.from_numpy(qis_img.astype(np.float32)/255.).permute(2, 0, 1)
        # qisden_img = torch.from_numpy(qisden_img.astype(np.float32)/255.).permute(2, 0, 1)
        # x_anchor_img = torch.from_numpy(x_anchor_img.astype(np.float32)/255.).permute(2, 0, 1)
        gt_img = torch.from_numpy(gt_img.astype(np.float32)/255.).permute(2, 0, 1)
        gt_gray_img = torch.from_numpy(gt_gray_img.astype(np.float32)/255.).permute(2, 0, 1)

        # prompt = metadata['struct_color_prompt_gt']
        prompt = ''
        gain = torch.from_numpy(np.array([metadata['iso_gain']]))
        # print('gain: ', gain, type(gain))
        lux = metadata['lux']

        # Normalize to [-1, 1]
        data = {
            'blur_1': cmos_img * 2 - 1,  # 0,1 to -1,1
            'blur_2': qis_img * 2 - 1,        # 0,1 to -1,1
            # 'QISDEN': qisden_img * 2 - 1,  # 0,1 to -1,1
            # 'x_anchor': x_anchor_img * 2 - 1,
            'gt_1': gt_img * 2 - 1,       # 0,1 to -1,1
            'gt_2': gt_gray_img * 2 - 1, # 0,1 to -1,1
            'qis_iso_gain': gain,
            'struct_color_desc': prompt, 
            'lux': lux
        }
        
        return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create dictionary of options
    lux = 1.0
    opt = {
        'folder': '/scratch/gilbreth/pchennur/qis_cmos_lowlight/simulated_dataset/val/',
        'val_lux': lux, 
    }
    
    # Create dataset
    dataset = CMOS_QIS_dataset_predefined(opt, mode='val')
    print(f"Dataset size: {len(dataset)}")
    
    # # Test loading a few samples
    # num_samples_to_test = min(10, len(dataset))
    
    for idx in range(5, len(dataset)):
        print(f"\nLoading sample {idx}...")
        sample = dataset[idx]
        
        cmos = sample['CMOS_blurry']
        qis = sample['QIS']
        gt = sample['GT_RGB']
        prompt = sample['struct_color_desc']
        # print(prompt)
        
        print(f"  CMOS shape: {cmos.shape}, min: {cmos.min():.4f}, max: {cmos.max():.4f}")
        print(f"  QIS shape: {qis.shape}, min: {qis.min():.4f}, max: {qis.max():.4f}")
        print(f"  GT shape: {gt.shape}, min: {gt.min():.4f}, max: {gt.max():.4f}")
        
        # Check for NaN values
        assert not torch.isnan(cmos).any(), "CMOS contains NaN"
        assert not torch.isnan(qis).any(), "QIS contains NaN"
        assert not torch.isnan(gt).any(), "GT contains NaN"
        
        print(f"  ✓ Sample {idx} loaded successfully!")
        
        # Visualize the first sample
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize from [-1, 1] to [0, 1]
        cmos_vis = (cmos.permute(1, 2, 0).numpy() + 1) / 2
        qis_vis = (qis.permute(1, 2, 0).numpy() + 1) / 2
        gt_vis = (gt.permute(1, 2, 0).numpy() + 1) / 2
        
        axes[0].imshow(np.clip(cmos_vis, 0, 1))
        axes[0].set_title('CMOS Simulated')
        axes[0].axis('off')
        
        axes[1].imshow(np.clip(qis_vis, 0, 1), cmap = 'gray')
        axes[1].set_title('QIS Simulated')
        axes[1].axis('off')
        
        axes[2].imshow(np.clip(gt_vis, 0, 1))
        axes[2].set_title('Ground Truth RGB')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('cmos_qis_dataset_test.png', dpi=100, bbox_inches='tight')
        print(f"  Visualization saved to 'cmos_qis_dataset_test.png'")
        plt.show()
        break
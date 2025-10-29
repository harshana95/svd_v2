import importlib
import os
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import yaml
from dataset import create_dataset
from utils.misc import DictAsMember, find_attr
from models.archs import _arch_modules, define_network
from torch.utils.data import DataLoader
from accelerate import Accelerator


model_name = "NAFNet_arch"
model_path = "/scratch/gilbreth/wweligam/experiments/SVD_Real/NAFNetRealColor 2025 09 11 23.28.33/checkpoint-6000/NAFNet_arch_1/"
dataset_1_name = "harshana95/quadratic_color_psfs_5db_updated_real_hybrid_Flickr2k_gt_v2_PCA_interp_file"

with open("/depot/chan129/users/harshana/svd_v2/checkpoints/ICASSP_26/common_archs/NAFNet_real_color.yml", mode='r') as f:
    data = OmegaConf.load(f)
    data = OmegaConf.to_yaml(data, resolve=True)
    data = yaml.safe_load(data)
    if isinstance(data, dict):
        opt = DictAsMember(**data)
        opt.pretty_print()
    else:
        opt = data
    
dataset = create_dataset(opt.datasets.val)

dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=1,
    num_workers=4,
)

c = find_attr(_arch_modules, model_name)
model = c.from_pretrained(model_path)
print(f"Loaded {model_name} model from {model_path}")

model.eval()
model = model.cuda()  # always run the model in GPU

with torch.no_grad():
    idx = 0
    for batch in dataloader:
        img = batch['blur'].cuda()
        # img = torch.flip(img, dims=[1])  # rgb to bgr
        out = model(img)
        print(img.min(), img.max())
        print(out.min(), out.max())

        plt.imsave(f"in_{idx}.png", np.clip(img[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        plt.imsave(f"out_{idx}.png", np.clip(out[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        idx += 1

        if idx > 5:
            break
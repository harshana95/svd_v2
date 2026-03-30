from diffusers.utils.export_utils import export_to_video
import einops
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from ptlflow.utils import flow_utils
import torch
import yaml
from dataset import create_dataset
from utils.misc import DictAsMember
from torch.utils.data import DataLoader
from tqdm import tqdm
    
dataset = create_dataset(DictAsMember(**{
    'type': 'SyntheticMotionBlurDataset',
    'name' : '/depot/chan129/users/harshana/Datasets/Deconvolution/hybrid_Flickr2k_gt_v2/',
    'split':'train',
    'gt_key': 'image',
    'lq_key': 'blur',
    'skip_frames': 4,
    'meta_key': 'flow',
    'meta_include_kernel': True,
    'kernel_size': 64,
    'global_blur': True,
    'return_meta': True,
    'transforms': {
        'to_tensor': {'keys': ['gt_key']},
        'select_channels':{'keys': ['gt_key'], 'channels': [True, True, True, False]},
        'crop':{'keys': ['gt_key'], 'h': 1024, 'w': 1024},
        'resize':{'keys': ['gt_key'], 'h': 512, 'w': 512},        
    },
    'foreground_dataset': {
        'name': '/scratch/gilbreth/wweligam/dataset/foreground_images/foreground/',
        'split': 'train',
        'split_range': [99, 100],
        'transforms': {
            'to_tensor': {'keys': ['gt_key']},
            'crop':{'keys': ['gt_key'], 'h': 1024, 'w': 1024},
            'resize':{'keys': ['gt_key'], 'h': 512, 'w': 512},
        },
    }

}))

dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=1,
    num_workers=0,
)


with torch.no_grad():
    idx = 0
    for batch in tqdm(dataloader):
        blur = batch['blur']*0.5+0.5
        gt = batch['image']*0.5+0.5
        fg = batch['fg']*0.5+0.5
        bg = batch['bg']*0.5+0.5
        mask = batch['mask']
        flow = batch['flow']
        flow = einops.rearrange(flow[0], '(f n) h w -> f n h w', n=2)
        gt_flow_video = []
        for i in range(flow.shape[0]):
            gt_flow_video.append(flow_utils.flow_to_rgb(flow[i].to(torch.float32)))
        gt_flow_video =  torch.stack(gt_flow_video, dim=0).numpy().transpose([0, 2, 3, 1])
        # breakpoint()
        export_to_video(gt_flow_video, f"./results/of_{idx}.mp4", fps=15)
        plt.imsave(f"./results/blur_{idx}.png", np.clip(blur[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        plt.imsave(f"./results/gt_{idx}.png", np.clip(gt[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        plt.imsave(f"./results/fg_{idx}.png", np.clip(fg[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        plt.imsave(f"./results/bg_{idx}.png", np.clip(bg[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        plt.imsave(f"./results/mask_{idx}.png", np.clip(mask[0, 0].cpu().numpy(), 0, 1))
        plt.imsave(f"./results/kernel_fg_{idx}.png", np.clip(batch['flow_kernel_fg'][0].cpu().numpy(), 0, 1))
        plt.imsave(f"./results/kernel_bg_{idx}.png", np.clip(batch['flow_kernel_bg'][0].cpu().numpy(), 0, 1))
        idx += 1

        if idx > 5:
            break
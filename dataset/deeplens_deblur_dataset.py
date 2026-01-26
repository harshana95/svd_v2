import glob
import os
import random
import cv2
import numpy as np
import torch
from torchvision import transforms 
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import yaml

try:
    from utils.misc import DictAsMember
except:
    import sys
    from pathlib import Path
    current_file_path = Path(__file__).resolve()
    sys.path.append(str(current_file_path.parent.parent))

from utils import get_dataset_util
from utils.dataset_utils import DictWrapper, crop_arr
from utils.misc import DictAsMember

from deeplens import GeoLens
from deeplens.optics.psf import conv_psf_map

def apply_displacement(lens, displacement_arr):
    # print(f"Applying d {displacement_arr}")
    for i in range(len(lens.surfaces)):
        lens.surfaces[i].d = displacement_arr[i]


class DeepLensDeblurDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_key = opt.gt_key
        self.lq_key = opt.lq_key
        self.data_percent = opt.get('data_percent', [0,100])
        self.render_method = opt.get('render_method', 'psf_map')
        self.num_proc = opt.get('num_proc', 1)
        self.num_frames = opt.get('num_frames', 10)
        self.offset_power = opt.get('offset_power', 0.1)
        self.return_psfs = opt.get('return_psfs', False)
        if self.return_psfs:
            assert self.render_method == 'psf_map', "If you want to return psfs, render_method must be psf_map"

        # load gt image filenames
        self.dataset = load_dataset(opt.gt_dataset_path, drop_labels=True, split=f'train')
        self.dataset = self.dataset.rename_column('image', self.gt_key)
        self.selected_indices = list(range(int(len(self.dataset)*self.data_percent[0]/100), int(len(self.dataset)*self.data_percent[1]/100)))
        self.dataset = self.dataset.select(self.selected_indices)
        
        # setup simulator
        self.lens = GeoLens(filename=opt["lens_path"], device=None)
        self.lens.set_sensor(sensor_res=opt["sensor_res"], sensor_size=self.lens.sensor_size)
        self.original_d = torch.tensor([s.d for s in self.lens.surfaces])
        self.initialize_offset_d()

        # setup transformations
        self.dataset = self.setup_dataset(self.dataset, opt)

    def initialize_offset_d(self):
        self.offset_d = torch.zeros_like(self.original_d)
        for i in range(len(self.offset_d)-1):
            d = self.original_d[i+1] - self.original_d[i]
            self.offset_d[i+1] = self.offset_d[i] + random.random()*d*self.offset_power

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item[f'{self.gt_key}_path'] = f"{self.gt_key}-{idx}.png"
        item[f'{self.lq_key}_path'] = f"{self.lq_key}-{idx}.png"

        # initialize optics and gt img
        self.initialize_offset_d()
        img = item[self.gt_key]
        is_single_channel = False
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
            is_single_channel = True
        
        # simulate optics and render lq image
        all_images = []
        all_psfs = []
        for n in range(self.num_frames):
            apply_displacement(self.lens, self.original_d + self.offset_d*n/self.num_frames)
            if self.return_psfs:
                psf_map = self.lens.psf_map_rgb(grid=10, ks=64, depth=-20000)
                img_render = conv_psf_map(img[None].to(self.lens.device), psf_map)
                blurred = img_render[0].cpu().clip(0,1)
                all_psfs.append(psf_map)
            else:     
                blurred = self.lens.render(img[None].to(self.lens.device), method=self.render_method)[0].cpu().clip(0,1)
            all_images.append(blurred)
        
        item[self.lq_key] = torch.stack(all_images)
        if self.return_psfs:
            item['psfs'] = torch.stack(all_psfs)
        if is_single_channel:
            item[self.lq_key] = item[self.lq_key][1:2] # select green channel only
        return item

    def setup_dataset(self, dataset, opt):
        all_transforms = []
        gt_key, lq_key = self.gt_key, self.lq_key
        
        for transform_name in opt.transforms:
            transform_opt = opt.transforms[transform_name]
            print(transform_opt)
            if transform_opt is None:
                transform_opt = {}
            keys = transform_opt.pop('keys', None)
            keys_new = transform_opt.pop('keys_new', None)
            if keys is None:
                keys = opt.common_transforms_keys
            if keys_new is None:
                keys_new = []
            keys = [key.replace("gt_key", gt_key) for key in keys]
            keys = [key.replace("lq_key", lq_key) for key in keys]
            keys_new = [key.replace("gt_key", gt_key) for key in keys_new]
            keys_new = [key.replace("lq_key", lq_key) for key in keys_new]
            
            transform = get_dataset_util(transform_name, transform_opt)
            all_transforms.append(DictWrapper(transform, keys, keys_new))
        print("All transformations for the dataset ", opt['type'])
        for tr in all_transforms:
            print(tr.f)
        all_transforms = transforms.Compose(all_transforms)
        def apply_transforms(batch):
            return all_transforms(batch)
        
        dataset = dataset.map(apply_transforms, num_proc=self.num_proc, batched=True, batch_size=4, load_from_cache_file=False if opt.skip_cache else True, keep_in_memory=False)
        dataset.set_format('pt')
        return dataset

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import einops
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        data = yaml.safe_load(f)
        if isinstance(data, dict):
            opt = DictAsMember(**data)
        else:
            opt = data

        
    dataset = DeepLensDeblurDataset(opt.dataset)
    print(len(dataset))
    
    for i in range(3):
        batch = dataset[i]
        print(batch['gt'].shape, batch['blur'].shape)
        plt.imsave(f'gt_{i}.png', einops.rearrange(batch['gt'], 'c h w -> h w c').numpy())
        blurred = einops.rearrange(batch['blur'], 'n c h w -> n h w c').numpy()
        # print(blurred.min(), blurred.max())
        for j in range(len(blurred)):
            plt.imsave(f'lq_{i}_{j}.png', blurred[j])


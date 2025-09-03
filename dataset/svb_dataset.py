import glob
import os
import cv2
import numpy as np
import torch
from torchvision import transforms 
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from utils import get_dataset_util
from utils.dataset_utils import DictWrapper, crop_arr


class SVBDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_key = opt.gt_key
        self.lq_key = opt.lq_key
        self.basis_coef = torch.from_numpy(opt.basis_coef).to(torch.float32)
        self.basis_psfs = torch.from_numpy(crop_arr(opt.basis_psfs, *self.basis_coef.shape[-2:])).to(torch.float32)
        # # load gt image filenames
        # _dataset_path = os.path.join(opt.gt_dataset_path, opt.gt_dataset_file_pattern)
        # dataset = np.array(sorted(glob.glob(_dataset_path)))
        # print(f"Number of images in {_dataset_path} = {len(dataset)} using {opt.data_percent}%")
        # dataset = dataset[:int(len(dataset) * opt.data_percent / 100)]

        self.dataset = load_dataset(opt.gt_dataset_path, drop_labels=True, split=f'train')
        self.dataset = self.dataset.rename_column('image', self.gt_key)
        self.selected_indices = list(range(int(len(self.dataset)*opt.data_percent[0]/100), int(len(self.dataset)*opt.data_percent[1]/100)))
        self.dataset = self.dataset.select(self.selected_indices)
        
        # print(self.dataset)
        # print(self.dataset.__getitem__(0))
        self.dataset = self.setup_dataset(self.dataset, opt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item[f'{self.gt_key}_path'] = f"{self.gt_key}-{idx}.png"
        item[f'{self.lq_key}_path'] = f"{self.lq_key}-{idx}.png"
        return item

    def setup_dataset(self, dataset, opt):
        all_transforms = []
        gt_key, lq_key = self.gt_key, self.lq_key
        for transform_name in opt.get('transforms'):
            transform_opt = opt.transforms[transform_name]
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
            if transform_name == "sv_convolution":
                transform_opt['basis_psfs'] = self.basis_psfs
                transform_opt['basis_coef'] = self.basis_coef
            transform = get_dataset_util(transform_name, transform_opt)
            all_transforms.append(DictWrapper(transform, keys, keys_new))
        print("All transformations for the dataset ", opt['type'])
        for tr in all_transforms:
            print(tr.f)
        all_transforms = transforms.Compose(all_transforms)
        def apply_transforms(batch):
            return all_transforms(batch)
        
        dataset = dataset.map(apply_transforms, num_proc=opt.num_proc, batched=True, batch_size=16, load_from_cache_file=False if opt.skip_cache else True, keep_in_memory=False)
        dataset.set_format('pt')
        return dataset
import numpy as np
import torch
from torchvision import transforms 
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from utils import get_dataset_util
from utils.dataset_utils import DictWrapper


class HuggingFaceDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        dataset_name, split = opt['name'], opt['split']
        self.gt_key = opt.gt_key
        self.lq_key = opt.lq_key
        self.dataset = load_dataset(dataset_name, split=split, trust_remote_code=opt.get('trust_remote_code', None))
        self.dataset = self.setup_dataset(self.dataset, opt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
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
            
            transform = get_dataset_util(transform_name, transform_opt)
            all_transforms.append(DictWrapper(transform, keys, keys_new))
        print("All transformations for the dataset ")
        for i, tr in enumerate(all_transforms):
            print(tr.f, tr.keys, tr.new_keys)

        # Break everything after apply_model to post process. This is done to enable batched processing of apply_model
        post_transforms = []
        post_proccess = False
        for i, tr in enumerate(all_transforms):
            if tr.f.__class__.__name__ == 'apply_model':
                post_transforms = all_transforms[i:]
                all_transforms = all_transforms[:i]
                post_proccess = True
                break
                
        all_transforms = transforms.Compose(all_transforms)
        post_transforms = transforms.Compose(post_transforms)

        # adding batch size here will iterate through rows one-by-one and apply to the table as batches
        dataset = dataset.map(all_transforms, 
                              num_proc=opt.num_proc if opt.num_proc is not None else 4, # if stuck use accelerate launch instead of python to run the script
                              batched=True,
                              batch_size=opt.map_batch_size if opt.map_batch_size is not None else 4,
                              load_from_cache_file=False if opt.skip_cache else True, 
                              keep_in_memory=False,
                              writer_batch_size=3000)
        dataset.set_format('pt')
        if post_proccess:
            # batch before applying transforms. TODO: meed to perform unbatch after this transform
            # dataset = dataset.batch(num_proc=16, batch_size=opt.map_batch_size if opt.map_batch_size is not None else 16)
            dataset = dataset.map(post_transforms, 
                                num_proc=1, # forced to 1 process
                                batched=True,
                                batch_size=opt.map_batch_size if opt.map_batch_size is not None else 4,
                                load_from_cache_file=False if opt.skip_cache else True, 
                                keep_in_memory=False,
                                writer_batch_size=3000)
            dataset.set_format('pt')
        return dataset
    
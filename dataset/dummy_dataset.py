import numpy as np
import torch
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize


class DummyDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.len = opt['size']
        self.key = 'x'
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {self.key: idx}

    
import importlib
import numpy as np
import random
import torch
import torch.utils.data
from functools import partial
from os import path as osp
import os

from utils import scandir, find_attr

__all__ = ['create_dataset']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in os.listdir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'dataset.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamic instantiation

    dataset_cls = find_attr(_dataset_modules, dataset_type)
    dataset = dataset_cls(dataset_opt)
    return dataset

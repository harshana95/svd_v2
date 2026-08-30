import importlib
import logging
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

def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamic instantiation
    dataset_cls = None
    # Iterate through module names, import one by one, and check for the class.
    for file_name in dataset_filenames:
        try:
            module = importlib.import_module(f'dataset.{file_name}')
        except Exception as e:
            # Some dataset modules depend on packages that aren't installed in
            # every environment (e.g. the DeepLens datasets need the
            # deeplens/GeoLens physics package). Skip modules that fail to
            # import instead of crashing every run regardless of dataset_type.
            logging.getLogger(__name__).warning(
                f"Skipping dataset.{file_name}: failed to import ({e})"
            )
            continue
        cls_ = getattr(module, dataset_type, None)
        if cls_ is not None:
            dataset_cls = cls_
            break

    if dataset_cls is None:
        raise ValueError(f"Dataset class '{dataset_type}' not found in any of the scanned modules.")

    dataset = dataset_cls(dataset_opt)
    return dataset

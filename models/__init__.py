# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
from os import path as osp

from utils import scandir, find_attr

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [v.replace('/', '.')[:-3] for v in scandir(model_folder, recursive=True) if v.endswith('_model.py')]

def create_model(opt, logger):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    model_cls = None
    # Iterate through module names, import one by one, and check for the class.
    for file_name in model_filenames:
        module = importlib.import_module(f'models.{file_name}')
        cls_ = getattr(module, model_type, None)
        if cls_ is not None:
            model_cls = cls_
            break

    if model_cls is None:
        raise ValueError(f"Model class '{model_type}' not found in any of the scanned modules.")
    model = model_cls(opt, logger)
    return model

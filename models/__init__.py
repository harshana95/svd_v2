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
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]

def create_model(opt, logger):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    model_cls = find_attr(_model_modules, model_type)
    model = model_cls(opt, logger)
    return model



# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import logging
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
        try:
            module = importlib.import_module(f'models.{file_name}')
        except Exception as e:
            # Some model families depend on packages that aren't installed in
            # every environment (e.g. DeepLensModels needs the deeplens/GeoLens
            # physics package, only present on some clusters). Skip modules
            # that fail to import instead of crashing every run regardless of
            # the requested model_type.
            # Use the stdlib logger, not the `logger` param: it's typically
            # accelerate.logging.get_logger(), which requires an Accelerator/
            # PartialState to already exist -- but create_model() runs before
            # model.prepare() creates one.
            logging.getLogger(__name__).warning(
                f"Skipping models.{file_name}: failed to import ({e})"
            )
            continue
        cls_ = getattr(module, model_type, None)
        if cls_ is not None:
            model_cls = cls_
            break

    if model_cls is None:
        raise ValueError(f"Model class '{model_type}' not found in any of the scanned modules.")
    model = model_cls(opt, logger)
    return model

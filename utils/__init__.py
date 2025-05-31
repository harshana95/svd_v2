from .misc import (
    scandir,
    initialize,
    find_attr,
    ordered_yaml, 
    generate_folder, 
    keep_last_checkpoints,
    log_image,
    log_metrics,
    get_basename
)

import utils.dataset_utils as dataset_utils

__all__ = [
    'scandir',
    'initialize',
    'find_attr',
    'ordered_yaml',
    'generate_folder',
    'keep_last_checkpoints',
    'log_image',
    'log_metrics',
    'get_basename'
    ]


def get_dataset_util(util_name, util_opt):
    """
    
    """
    util_cls = getattr(dataset_utils, util_name, None)
    if util_cls is None:
        raise ValueError(f"{util_name} not found in dataset_utils.py")
    util = util_cls(**util_opt)
    return util

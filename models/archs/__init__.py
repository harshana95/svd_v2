import importlib
import logging
from os import path as osp

from utils import scandir, find_attr

# automatically scan and import arch modules
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [v.replace('/', '.')[:-3] for v in scandir(arch_folder, recursive=True) if v.endswith('_arch.py')]

def find_network_class(network_type):
    config_class = None
    net_class = None
    if network_type[-5:] == '_arch':
        network_type = network_type[:-5]
    # Iterate through module names, import one by one, and check for the classes.
    for file_name in arch_filenames:
        try:
            module = importlib.import_module(f'models.archs.{file_name}')
        except Exception as e:
            # Some archs depend on packages that aren't installed in every
            # environment. Skip modules that fail to import instead of
            # crashing every run regardless of the requested network_type.
            logging.getLogger(__name__).warning(
                f"Skipping models.archs.{file_name}: failed to import ({e})"
            )
            continue
        cls_ = getattr(module, network_type + '_arch', None)
        if cls_ is not None:
            net_class = cls_
            config_class = getattr(module, network_type + '_config', None)
            break

    if net_class is None:
        raise ValueError(f"Network class '{network_type}_arch' not found in any of the scanned modules.")
    return net_class, config_class

def define_network(opt):
    network_type = opt.pop('type')

    net_class, config_class = find_network_class(network_type)

    if config_class is not None:
        config = config_class(**opt.__dict__)
        net = net_class(config)
    else:
        from utils.misc import DictAsMember
        if isinstance(opt, DictAsMember):
            net = net_class(**opt.__dict__)
        else:
            net = net_class(**opt)
        
        
    return net

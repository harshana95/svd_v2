import importlib
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
        module = importlib.import_module(f'models.archs.{file_name}')
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

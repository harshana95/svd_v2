import importlib
from os import path as osp

from utils import scandir, find_attr

# automatically scan and import arch modules
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [v.replace('/', '.')[:-3] for v in scandir(arch_folder, recursive=True) if v.endswith('_arch.py')]
_arch_modules = [importlib.import_module(f'models.archs.{file_name}') for file_name in arch_filenames]


def define_network(opt):
    network_type = opt.pop('type')
    # initialize the config class
    config = find_attr(_arch_modules, network_type+'_config')(**opt)
    # initialize the network class
    net = find_attr(_arch_modules, network_type+'_arch')(config)
    return net

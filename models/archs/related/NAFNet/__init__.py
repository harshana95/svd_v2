# # ------------------------------------------------------------------------
# # Copyright (c) 2022 megvii-model. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from BasicSR (https://github.com/xinntao/BasicSR)
# # Copyright 2018-2020 BasicSR Authors
# # ------------------------------------------------------------------------
# import importlib
# from os import path as osp

# from utils import scandir, find_attr

# # automatically scan and import arch modules
# # scan all the files under the 'archs' folder and collect files ending with
# # '_arch.py'
# arch_folder = osp.dirname(osp.abspath(__file__))
# arch_filenames = [
#     osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
#     if v.endswith('_arch.py')
# ]
# # import all the arch modules
# _arch_modules = [
#     importlib.import_module(f'models.archs.related.NAFNet.{file_name}')
#     for file_name in arch_filenames
# ]


# def dynamic_instantiation(modules, cls_type, opt):
#     """Dynamically instantiate class.

#     Args:
#         modules (list[importlib modules]): List of modules from importlib
#             files.
#         cls_type (str): Class type.
#         opt (dict): Class initialization kwargs.

#     Returns:
#         class: Instantiated class.
#     """
#     # initialize the config class
#     config = find_attr(modules, cls_type+'_config')(**opt)

#     # initialize the network class
#     net = find_attr(modules, cls_type+'_arch')(config)

#     return net



# def define_network(opt):
#     network_type = opt.pop('type')
#     net = dynamic_instantiation(_arch_modules, network_type, opt)
#     return net

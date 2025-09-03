import matplotlib
import yaml

from dataset import create_dataset
from utils.misc import DictAsMember, get_basename
matplotlib.use("Agg")

import argparse
import glob
import os
import shutil
import cv2
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from psf.svpsf import PSFSimulator
from utils import generate_folder
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--comment', type=str, help='')
    parser.add_argument('--psf_data_path', type=str, help='')
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        data = yaml.safe_load(f)
        if isinstance(data, dict):
            opt = DictAsMember(**data)
        else:
            opt = data
        # Loader, _ = ordered_yaml()
        # opt = yaml.load(f, Loader=Loader)
        
    dataset_name = f"synthetic_{get_basename(opt.dataset.gt_dataset_path)}_{get_basename(args.psf_data_path)}_{args.comment}"

    # initialize save locations
    print(f"Saving to {join(opt.save_dir, dataset_name)}", "Dataset name", dataset_name)
    ds_dir = join(opt.save_dir, dataset_name)
    gt_dir = join(ds_dir, 'train', 'gt')
    meas_dir = join(ds_dir, 'train', 'blur')
    vgt_dir = join(ds_dir, 'val', 'gt')
    vmeas_dir = join(ds_dir, 'val', 'blur')
    md_dir = join(ds_dir, 'metadata')

    generate_folder(gt_dir)
    generate_folder(meas_dir)
    generate_folder(vgt_dir)
    generate_folder(vmeas_dir)
    generate_folder(md_dir)

    # save args to log
    with open(join(md_dir, 'args.txt'), 'w') as f:
        for key, value in vars(opt).items():
            f.write(f'{key}: {value}\n')
            f.flush()
        f.flush()

    # load PSFs and simulator
    psf_ds, metadata = PSFSimulator.load_psfs(args.psf_data_path, "PSFs_with_basis.h5")
    image_shape = psf_ds['H_img'].shape[:2]
    psf_size = psf_ds['psfs'].shape[-1]

    psfsim = PSFSimulator(image_shape, psf_size=psf_size,  normalize=False)
    psfsim.set_H_obj(homography=psf_ds['H_obj'][:])
    psfsim.set_H_img(homography=psf_ds['H_img'][:])
    all_psfs = psf_ds['psfs'][:]
    opt.dataset['basis_psfs'] = psf_ds['basis_psfs'][:]
    opt.dataset['basis_coef'] = psf_ds['basis_coef'][:]
    psf_ds.close()
    ds = create_dataset(opt.dataset)

    sample = ds.__getitem__(0)
    print(f"################## Keys and value shapes of the dataset")
    for key in sample.keys():
        try:
            print(f"{key} {sample[key].shape} {sample[key].dtype}")
        except:
            print(key, sample[key])

    # ds_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1)
    # ================================================================================ Iterate through the dataset
    idx = 0
    size = len(ds)
    ds = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    for data in tqdm(ds, 'Generating images'):
        _x = data[opt.dataset.gt_key].cpu().numpy()
        _y = data[opt.dataset.lq_key].cpu().numpy()
        for i in range(len(_x)):
            x = _x[i].transpose([1, 2, 0])
            y = _y[i].transpose([1, 2, 0])
            dir_gt, dir_meas = gt_dir, meas_dir
            if idx < size * opt.val_perc / 100:
                dir_gt, dir_meas = vgt_dir, vmeas_dir

            cv2.imwrite(os.path.join(dir_gt, f'{idx:05d}.png'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR) * 255)
            cv2.imwrite(os.path.join(dir_meas, f'{idx:05d}.png'), cv2.cvtColor(y, cv2.COLOR_RGB2BGR) * 255)
            idx += 1

    # ================================================================================================== pushing to HF
    from datasets import disable_caching
    from huggingface_hub import HfApi

    disable_caching()

    shutil.copyfile('./dataset/loading_script.py', join(ds_dir, f'{dataset_name}.py'))
    shutil.rmtree('./.cachehf', ignore_errors=True)
    dataset = load_dataset(ds_dir, trust_remote_code=True, cache_dir='./.cachehf')
    shutil.rmtree('./.cachehf', ignore_errors=True)
    print(f"Length of the created dataset {len(dataset)}")

    repoid = f"harshana95/{dataset_name}"
    dataset.push_to_hub(repoid, num_shards={'train': 100, 'val': 1})

    api = HfApi()
    api.upload_folder(
        folder_path=md_dir,
        repo_id=repoid,
        path_in_repo="metadata",
        repo_type="dataset",

    )


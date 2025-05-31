import matplotlib
from tqdm import tqdm
matplotlib.use("TkAgg")

import argparse
import os
import cv2
from psf.svpsf import PSFSimulator
from utils.dataset_utils import crop_arr

from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from utils.pca_utils import get_pca_components

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--psf_data_path', default="/scratch/gilbreth/wweligam/dataset/gaussian_synthetic")

    parser.add_argument('--psf_decomp', type=str, default='PCA', choices=["PCA", "sections", "Zernike", "Uniform"])
    parser.add_argument('--interpolation_samples', type=int, default=128)

    parser.add_argument('--pca_n', type=int, default=50)
    parser.add_argument('--zernike_n', type=int, default=15)

    parser.add_argument('--normalize_psf', action='store_true')

    args = parser.parse_args()

    psf_decomp = args.psf_decomp

    # load PSFs and simulator
    psf_ds, metadata = PSFSimulator.load_psfs(args.psf_data_path, "psfs.h5")
    image_shape = psf_ds['H_img'].shape[:2]
    psf_size = psf_ds['psfs'].shape[-1]
    psfsim = PSFSimulator(image_shape, psf_size=psf_size,  normalize=args.normalize_psf)
    psfsim.set_H_obj(homography=psf_ds['H_obj'][:])
    psfsim.set_H_img(homography=psf_ds['H_img'][:])
    all_psfs = psf_ds['psfs'][:]
    psf_ds.close()

    # ========================================================================================= generate PSF basis
    if psf_decomp == "PCA":
        sample_shape = (args.interpolation_samples, args.interpolation_samples)
        PSFSimulator.display_psfs(image_shape=image_shape, psfs=all_psfs, weights=None, metadata=metadata['psf_metadata'], title="Loaded PSFs", skip=1)
        plt.savefig(join(args.psf_data_path, 'psfs_all.png'))

        # fit PCA and create weights according to PCA coefficients
        basis_psfs, basis_coef, pca_components, pca_mean, pca_var = get_pca_components(all_psfs, args.pca_n)
        a, b, c = basis_coef.shape[:3]
        weights_resized = np.zeros((a, b, c, image_shape[0], image_shape[1]))
        for i in tqdm(range(a), desc="resizing weights"):
            for j in range(b):
                weights_resized[i, j] = cv2.resize(basis_coef[i, j].transpose([1, 2, 0]),
                                                   (image_shape[1], image_shape[0]),
                                                   interpolation=cv2.INTER_LINEAR, ).transpose([2, 0, 1])
        basis_coef = weights_resized
        psfs_img = PSFSimulator.display_psfs(image_shape, basis_psfs, basis_coef, skip=image_shape[0] // 10)
        plt.savefig(join(args.psf_data_path, 'psfs_sampled_from_basis.png'))
        plt.cla()
    # =================================================================== Generate PSFs weighted by sectors of the image
    elif psf_decomp == "sections":
        raise NotImplementedError()
    # =================================================================== Generate PSFs weighted by Zernike coefficients
    elif psf_decomp == "Zernike":
        n = args.zernike_n
        raise NotImplementedError()
    else:
        raise Exception("invalid psf decomposition type")
    # ======================================================================================= End PSF generation

    # ============================================================================== pad the psfs to get the image shape
    print(f"Original PSF/Weight size {basis_coef.shape[-2:]}")

    # ============================================================================= Create the dataset Synthetic data
    PSFSimulator.save_psfs(args.psf_data_path, all_psfs, metadata=metadata,
                           basis_psfs=basis_psfs, basis_coef=basis_coef,
                           H_img=psfsim.H_img, H_obj=psfsim.H_obj, save_as_image=True, fname="PSFs_with_basis.h5")
    
    
    psfs, metadata = psfsim.load_psfs(args.psf_data_path, fname="PSFs_with_basis.h5")
    for name in psfs:
        print(f"{name} {psfs[name].shape}")

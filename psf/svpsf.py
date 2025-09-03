import os
import random
from os.path import join
import h5py
import cv2
import einops
import numpy as np
import matplotlib.pyplot as plt
import scipy
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from zernike import RZern
from deeplens import GeoLens

try:
    from utils import get_dataset_util
    from utils.dataset_utils import DictWrapper, crop_arr
    from psf.motionblur.kernels import KernelDataset
    from utils import generate_folder
except:
    import sys
    from pathlib import Path
    current_file_path = Path(__file__).resolve()
    sys.path.append(str(current_file_path.parent.parent))

    from utils import get_dataset_util
    from utils.dataset_utils import DictWrapper, crop_arr
    from psf.motionblur.kernels import KernelDataset
    from utils import generate_folder

def normal_index_func(_i, _s):
    return _i - (_s - 1) / 2


def rand_index_func(_i, _s):
    return (np.random.rand() - 0.5) * _s


def sigma_scales_elliptical(x, y, r):
    return 1, ((x * x + y * y) ** 0.5) / r


def sigma_scales_circular(x, y, r):
    d = ((x * x + y * y) ** 0.5) / r
    return d, d


def apply_Homography(x, y, H):
    hc = np.array([[x, y, 1]]).T
    new_hc = H @ hc
    new_x, new_y = new_hc[0] / new_hc[2], new_hc[1] / new_hc[2]
    if new_x.shape[0] == 1:
        new_x = new_x[0]
        new_y = new_y[0]
    return new_x, new_y


class PSFSimulator:
    def __init__(self, image_shape, psf_size=11, normalize=True, wavelengths=[472.5, 532.8, 685]):
        """
        @param image_shape: image shape should be the corresponding shape of the image on sensor
        @param psf_size:
        @param normalize:
        """
        self.image_shape = np.array(image_shape)
        self.psf_size = psf_size
        self.normalize = normalize
        self.wavelengths = wavelengths

        self.cart = None
        self.L = None

        self._H_obj = None  # mapping from index space to object space (arbitrary position in space in mm)
        self._H_img = None  # mapping from index space to image space (pixel position)

        # motion kernel used for generating motion blur
        self.motion = KernelDataset(psf_size, psf_size // 2)

        # lens model for deeplens
        self.lens = None
        self.lens_path = None
        
        h, w = self.image_shape
        self.R = 0.5 * ((h * h + w * w) ** 0.5)

        if normalize:
            print("PSFs will be normalized")
        else:
            print("PSFs will NOT be normalized")

    @property
    def H_obj(self):
        if self._H_obj is None:
            raise ValueError("Set H_obj homography (mapping from index space to object space) first!")
        return self._H_obj

    @property
    def H_img(self):
        if self._H_img is None:
            raise ValueError("Set H_img homography (mapping from index space to image space) first!")
        return self._H_img

    def set_H_obj(self, homography=None, **kwargs):
        if homography is not None:
            assert len(homography.shape) == 4
            assert homography.shape == (*self.image_shape, 3, 3)
            self._H_obj = homography
        else:
            h = np.eye(3)
            h[0, 0] = kwargs.get("horizontal_distance")
            h[1, 1] = kwargs.get("vertical_distance")
            h[0, 2] = kwargs.get("horizontal_translation")
            h[1, 2] = kwargs.get("vertical_translation")
            h = h[None, None]
            h = np.repeat(h, self.image_shape[0], 0)
            h = np.repeat(h, self.image_shape[1], 1)
            self._H_obj = h

    def set_H_img(self, homography=None, **kwargs):
        if homography is not None:
            assert len(homography.shape) == 4
            assert homography.shape == (*self.image_shape, 3, 3)
            self._H_img = homography
        else:
            h = np.eye(3)
            h[0, 0] = kwargs.get("horizontal_distance")
            h[1, 1] = kwargs.get("vertical_distance")
            h[0, 2] = kwargs.get("horizontal_translation")
            h[1, 2] = kwargs.get("vertical_translation")
            h = h[None, None]
            h = np.repeat(h, self.image_shape[0], 0)
            h = np.repeat(h, self.image_shape[1], 1)
            self._H_img = h

    def generate_psf(self, x, y, ch, **kwargs):
        """
        Generate a PSF centered at (x, y) with specified parameters.

        @param x: float (x location in image [-width/2  width/2]) in mm
        @param y: float (y location in image [-height/2 height/2]) in mm
        @param ch: int (channel of the image)

        @return psf, metadata: np.array (psf with shape [psf_size, psf_size]), metadata of the PSF
        """
        metadata = []
        psf_type = kwargs.get('psf_type')
        xx_offset, yy_offset = 0, 0
        chromatic = kwargs.get('chromatic', [0,0,0])[ch]
        chromatic_trans = kwargs.get('chromatic_trans', [1, 1, 1])[ch]
        if psf_type == 'random':
            psf_type = random.choice(['gaussian', 'motion'])

        # setup constants
        if psf_type == "gaussian" or psf_type == "disk": # xx, yy used for psf_type == 'gaussian' or psf_type == 'disk'
            if chromatic != 0:
                xx_offset = (x/self.R)*(self.psf_size/2)*chromatic
                yy_offset = (y/self.R)*(self.psf_size/2)*chromatic
                x += int(xx_offset)
                y += int(yy_offset)

            self.xx, self.yy = np.meshgrid(np.arange(self.psf_size, dtype='float32'), np.arange(self.psf_size, dtype='float32'))
            self.xx = self.xx - (self.psf_size / 2) + xx_offset
            self.yy = self.yy - (self.psf_size / 2) + yy_offset
            self.xx /= self.psf_size/2
            self.yy /= self.psf_size/2
        elif psf_type == 'deeplens':
            # setup simulator
            if self.lens is None or self.lens_path != kwargs['lens_path']:
                self.lens_path = kwargs['lens_path']
                self.lens = GeoLens(filename=kwargs["lens_path"])
                self.lens.double()
                self.lens.set_sensor(sensor_res=kwargs["sensor_res"], sensor_size=self.lens.sensor_size)
            
        # generate the PSF
        if psf_type == 'gaussian':
            if kwargs['gaussian_type'] == 'elliptical':
                s1_scale, s2_scale = sigma_scales_elliptical(x, y, self.R)
            elif kwargs['gaussian_type'] == 'circular':
                s1_scale, s2_scale = sigma_scales_circular(x, y, self.R)
            else:
                raise Exception("Wrong Gaussian type")
            sigma1 = kwargs['sigma1'] * s1_scale * (1 + 0)
            sigma2 = kwargs['sigma2'] * s2_scale * (1 + chromatic)
            sigma1 = max(sigma1, kwargs.get('sigma_min'))
            sigma2 = max(sigma2, kwargs.get('sigma_min'))

            theta = np.arctan2(x, y) + np.pi / 2  # Fixed bug

            a = ((np.cos(theta) ** 2) / (2 * sigma2 ** 2)) + ((np.sin(theta) ** 2) / (2 * sigma1 ** 2))
            b = -((np.sin(2 * theta)) / (4 * sigma2 ** 2)) + ((np.sin(2 * theta)) / (4 * sigma1 ** 2))
            c = ((np.sin(theta) ** 2) / (2 * sigma2 ** 2)) + ((np.cos(theta) ** 2) / (2 * sigma1 ** 2))

            mu = [0, 0]
            A = 1
            psf = A * np.exp(-(a * (self.xx - mu[0]) ** 2 + 2 * b * (self.xx - mu[0]) * (self.yy - mu[1]) + c * (
                    self.yy - mu[1]) ** 2))
            metadata = [theta, sigma1, sigma2, x, y, xx_offset, yy_offset]
        elif psf_type == 'disk':
            sigma = kwargs.get('sigma', 0.1)
            psf = np.sqrt(self.xx ** 2 + self.yy ** 2) <= sigma
        elif psf_type == 'psf_n_weight':
            psfs = kwargs.get('basis_psfs')
            weights = kwargs.get('basis_coef')
            inc, outc, n_psfs, hp, wp = psfs.shape
            inc, outc, n_psfs, h, w = weights.shape
            i, j = int(x), int(y)
            psf = (psfs * weights[:, :, :, i:i + 1, j:j + 1]).sum(2)
        elif psf_type == 'motion':
            psf = self.motion.sample().astype(np.float32)
        elif psf_type == 'zernike':
            nz = kwargs.get('n')
            coeffN = (nz + 1) * (nz + 2) // 2
            if self.cart is None:
                self.cart = RZern(nz)
                ddx = np.linspace(-1.0, 1.0, self.psf_size)
                ddy = np.linspace(-1.0, 1.0, self.psf_size)
                xv, yv = np.meshgrid(ddx, ddy)
                rho = np.sqrt(np.square(xv) + np.square(yv))
                self.cart.make_cart_grid(xv, yv, unit_circle=True)

                D = 0.33
                strength = 4 / D
                R = np.diag(np.exp(-0.01 * np.linspace(0, 20, coeffN) ** 2)) * np.sqrt(strength)
                self.L = np.linalg.cholesky(R)
                self.mask = np.zeros_like(rho)
                self.mask[rho < D] = 1  # circular apperture

            noise = np.random.randn(1, coeffN)
            zvecs = noise @ self.L
            phase = self.cart.eval_grid(zvecs[0], matrix=True)
            phase = np.nan_to_num(phase)
            psf = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.ifftshift(self.mask * np.exp(-1j * phase)))) ** 2)
            psf = np.nan_to_num(psf)
            metadata = zvecs[0]
        elif psf_type == 'deeplens':
            x_norm, y_norm = x/(self.image_shape[1]//2), y/(self.image_shape[0]//2)  # Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            y_norm = -y_norm  # flip y axis
            psf = self.lens.psf([x_norm, y_norm, -10000.0], ks=self.psf_size, recenter=True, wvln=self.wavelengths[ch]/1000) # wvln in um
            psf = psf.cpu().numpy()
            metadata = [x, y, x_norm, y_norm]
        else:
            raise ValueError("Invalid PSF type.")
            
        psf *= chromatic_trans
        return psf, np.array(metadata)

    def generate_all_psfs(self, nh, nw, inc, outc, **kwargs):
        """
        Generate spatially varying PSFs for each pixel. Can use memory heavily!!!
        @param nh: int (number of points vertical)
        @param nw: int (number of points horizontal)
        @param inc: int (in channels)
        @param outc: int (out channels)

        @return psfs: np.array (psfs with shape [outc, inc, nh, nw, psf_size, psf_size])
        """
        print(f"Generating PSFs {nh}x{nw} channels {inc}")
        print(f"kwargs {kwargs}")
        # handle special case when uniform PSFs are needed
        if kwargs.get('is_uniform', False):
            psfs, metadata_arr = [], []
            for c in range(inc):
                psf, metadata = self.generate_psf(0, 0, c, **kwargs)
                psf = np.expand_dims(psf, 0)
                psf = np.expand_dims(psf, 0).repeat(outc, 0)
                psfs.append(psf)
                metadata_arr.append(metadata)
            psfs, metadata_arr = np.array(psfs), np.array(metadata_arr)
            weights = np.ones((inc, outc, 1, *self.image_shape))
            psfs_all = self.generate_pixelwise_psfs_from_weighted_psfs(psfs, weights)
            return psfs_all, metadata_arr

        # find the PSFs for each object position
        psfs = np.zeros((inc, outc, nh, nw, self.psf_size, self.psf_size))
        pbar = tqdm(total=nh * nw, desc="Creating PSFs")
        metadata_arr = None
        max_psf_size = [0, 0]
        position_indices = [(x, y) for x in range(nw) for y in range(nh)]
        for n in range(len(position_indices)):
            x, y = position_indices[n]
            x_obj, y_obj = apply_Homography(x, y, self.H_obj[x, y])
            x_img, y_img = apply_Homography(x, y, self.H_img[x, y])  # pixel position on image space
            x_img, y_img = int(x_img), int(y_img)
            assert 0 <= y_img <= self.image_shape[0]
            assert 0 <= x_img <= self.image_shape[1]

            # Generate the PSF and store it

            for i in range(inc):
                for j in range(outc):
                    # move origin to center for easy simulation. remove if this is bad for file
                    _x_img, _y_img = x_img - (self.image_shape[1] // 2), y_img - (self.image_shape[0] // 2)
                    psf, metadata = self.generate_psf(_x_img, _y_img, i, **kwargs)
                    psfs[i, j, y, x] = psf

                    max_psf_size[0] = max(max_psf_size[0], psf.shape[0])
                    max_psf_size[1] = max(max_psf_size[1], psf.shape[1])

                    metadata = np.concatenate([np.array([y_img, x_img]), metadata])
                    if metadata_arr is None:
                        metadata_arr = np.zeros((*psfs.shape[:-2], len(metadata)))
                    metadata_arr[i][j][y][x] = metadata
            if self.normalize:  # Normalize the PSF to have unit sum
                psf_sum = np.sum(psfs[:,:, y, x])
                if psf_sum == 0:
                    psf_sum = 1
                psfs[:,:, y, x] /= psf_sum/inc
            pbar.update()
            pbar.set_postfix_str(f"x={x:04d} y={y:04d}  meta={' '.join([f'{m:4.2f}' for m in metadata_arr[0,0,y,x, :].flatten()])}")
        print(f"Max psf size: {max_psf_size}")
        pbar.close()
        metadata_arr = np.array(metadata_arr).transpose([0, 1, 4, 2, 3])  # inc outc d h w
        return psfs, metadata_arr

    @staticmethod
    def generate_pixelwise_psfs_from_weighted_psfs(basis_psfs, weights, is_mean_ker=True):
        print(f"psfs {basis_psfs.shape} weights {weights.shape}")
        inc, outc, n_psfs, hp, wp = basis_psfs.shape
        h, w = weights.shape[-2], weights.shape[-1]

        psfs_all = np.zeros((inc, outc, h, w, hp, wp))
        with tqdm(total=h * w, desc="Generating all pixel PSFs from weighted PSFs") as pbar:
            for i in range(h):
                for j in range(w):
                    psfs_all[:, :, i, j] = (basis_psfs * weights[:, :, :, i:i + 1, j:j + 1]).sum(2)
                    pbar.update()
        return psfs_all

    @staticmethod
    def display_psfs(image_shape, psfs, weights=None, metadata=None, translate=(0, 0), q=0, skip=1, gap=(1, 1), title=""):
        k = psfs.shape[-1]
        if weights is not None:
            tmp = np.zeros((psfs.shape[0], image_shape[0] + k, image_shape[1] + k))
            for i in tqdm(range(0, image_shape[0], skip), desc="Displaying image"):
                for j in range(0, image_shape[1], skip):

                    try:
                        i *= gap[1]
                        j *= gap[0]
                        tmp[:, i:i + k, j:j + k] += (
                                psfs[:, q, :] * weights[:, q, :, i + translate[0], j + translate[1]].reshape(
                            [weights.shape[0], weights.shape[2], 1, 1])).sum(1)
                    except:
                        pass
            tmp = tmp[:, k // 2:-k // 2, k // 2:-k // 2]
        else:
            assert metadata is not None
            tmp = np.zeros((psfs.shape[0], image_shape[0] + k, image_shape[1] + k))
            for i in range(0, psfs.shape[2], skip):
                for j in range(0, psfs.shape[3], skip):
                    psf = psfs[:, q, i, j]
                    y_img, x_img = metadata[0, q, :2, i, j].astype(int)  # use from inc=0, first two elements of the metadata should be y_img, and x_img
                    try:
                        tmp[:, y_img:y_img + k, x_img:x_img + k] += psf
                    except:
                        print(f"Failed to plot {y_img}, {x_img} with k={k}")
            # tmp = einops.rearrange(psfs[:, q, ::skip, ::skip], "c H W h w -> c (H h) (W w)")
        print(f"Displaying PSFs {tmp.shape}")
        plt.imshow(tmp.transpose([1, 2, 0]) / tmp.max())
        plt.title(title + f" max value={tmp.max():.2f} min value={tmp.min():.2f}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        return tmp

    @staticmethod
    def save_psfs(path, psfs,
                  psfs_nopad=None, metadata=None,
                  basis_psfs=None, basis_coef=None,
                  H_obj=None, H_img=None,
                  save_as_image=False,
                  dtype='float16', fname="psfs.h5"):
        generate_folder(path)

        print("##################### Saving PSFs")
        hf = h5py.File(join(path, fname), "w")
        hf.create_dataset("psfs", data=psfs.astype(dtype))
        if metadata is not None:
            if isinstance(metadata, dict):
                for k in metadata:
                    if isinstance(metadata[k], np.ndarray):
                        print(f'{k} {metadata[k].shape}')
                        hf.create_dataset('metadata.'+k, data=metadata[k].astype(dtype))
                    else:
                        print(f'{k} {metadata[k]}')
                        hf.attrs[k] = metadata[k]
            else:
                print(f"Metadata {metadata.shape}")            
                hf.create_dataset("metadata", data=metadata.astype(dtype))

        if basis_psfs is not None:
            print(f"basis_psfs {basis_psfs.shape}")
            hf.create_dataset("basis_psfs", data=basis_psfs.astype(dtype))
        if basis_coef is not None:
            print(f"basis_coef {basis_coef.shape}")
            hf.create_dataset("basis_coef", data=basis_coef.astype(dtype))

        if psfs_nopad is not None:
            print(f"PSFs without padding {psfs_nopad.shape}")
            hf.create_dataset("psfs_no_pad", data=psfs_nopad.astype(dtype))


        if H_obj is not None:
            print(f"Hobj {H_obj.shape}")
            hf.create_dataset("H_obj", data=H_obj.astype(dtype))
        if H_img is not None:
            print(f"Himg {H_img.shape}")
            hf.create_dataset("H_img", data=H_img.astype(dtype))

        # hf.attrs.update(dict)
        hf.close()

        # if psfs_all is not None:  # this can be very large. so save separately
        #     print(f"All PSFs {psfs_all.shape}")
        #     hf = h5py.File(join(path, "PSFs_ALL.h5"), "w")
        #     hf.create_dataset("psfs", data=psfs_all.astype(dtype))
        #     hf.close()
        if save_as_image:
            generate_folder(join(path, 'basis_psfs'))
            if basis_coef is not None:
                generate_folder(join(path, 'basis_coef'))

            for i in range(basis_psfs.shape[0]):
                for j in range(basis_psfs.shape[1]):
                    for n in range(basis_psfs.shape[2]):
                        cv2.imwrite(join(path, 'basis_psfs', f'{i}_{j}_{n}.png'),
                                    128 + (basis_psfs[i, j, n] * 255).astype(int))
                        print(f'PSF {i}_{j}_{n}.png Min:{basis_psfs[i, j, n].min()} Max:{basis_psfs[i, j, n].max()}')
                        if basis_coef is not None:
                            cv2.imwrite(join(path, 'basis_coef', f'{i}_{j}_{n}.png'),
                                        128 + (basis_coef[i, j, n] * 255).astype(int))

    @staticmethod
    def load_psfs(path, fname="psfs.h5"):
        try:
            psfs = h5py.File(os.path.join(path, fname), "r")
        except:
            psfs = hf_hub_download(repo_id=path, filename=f"metadata/{fname}", repo_type="dataset")
            psfs = h5py.File(psfs, "r")
        metadata = {}
        for k in psfs.attrs.keys():
            metadata[k] = psfs.attrs[k]
        for k in psfs:
            if k.startswith('metadata.'):
                metadata[k[9:]] = psfs[k][:] # load data completely
        return psfs, metadata

    @staticmethod
    def convert_kernels_splat2conv(kernels, weights=None, normalize=True):
        if weights is None:
            unsq = False
            if len(kernels.shape) == 6:
                kernels = kernels[0]
                unsq = True
            print("Allocating space for scattering to gathering")
            c, H, W, h, w = kernels.shape
            _kernels = np.zeros_like(kernels)
            for i in tqdm(range(H), desc="Converting kernels from splat to conv"):
                for j in range(W):
                    for m in range(h):
                        for n in range(w):
                            _m = i - h // 2 + m
                            _n = j - w // 2 + n
                            if _m < 0 or _n < 0 or _m >= H or _n >= W: continue
                            _kernels[:, i, j, m, n] = kernels[:, _m, _n, h - m - 1, w - n - 1]

            # _kernels = np.pad(kernels, ((0, 0), (h // 2, h // 2), (w // 2, w // 2), (0, 0), (0, 0)), mode='constant')
            # for i in tqdm(range(H), desc="Converting kernels from splat to conv"):
            #     for j in range(W):
            #         _kernels[:, i, j] = np.flip(np.roll(_kernels[:, i, j], -h // 2 + 1, axis=0), axis=0)
            #         _kernels[:, i, j] = np.flip(np.roll(_kernels[:, i, j], -w // 2 + 1, axis=1), axis=1)
            # _kernels = _kernels[:, h // 2:-h // 2, w // 2:-w // 2]

            if normalize:
                print("Normalizing kernels")
                sums = np.sum(_kernels, axis=(-1, -2), keepdims=True)
                sums[sums == 0.] = 1
                _kernels /= sums
            if unsq:
                _kernels = np.expand_dims(_kernels, 0)
        else:
            raise Exception("Can't convert to conv with weights")

        return _kernels

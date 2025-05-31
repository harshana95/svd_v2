import shutil
import zipfile
import comet_ml
import argparse
import cv2
import torchvision
from tqdm import tqdm
import yaml
import os
import einops
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import numpy.core.numeric as NX
import torch
from datasets import load_dataset
from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction # Using the legacy version as it takes wavelength as a forward input instead of initialization input
from dflat.render import Fronto_Planar_Renderer_Incoherent

# from PySide2.QtCore import QLibraryInfo
from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

def polyder(p, m=1):
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")

    n = len(p) - 1
    y = p[:-1] * NX.arange(n, 0, -1)[:, None]
    if m == 0:
        val = p
    else:
        val = polyder(y, m - 1)
    return val

def crop_arr(arr, h, w, mode='constant'):  # todo: this is too slow
    hw, ww = arr.shape[-2:]
    do_pad = False
    istorch = type(arr) == torch.Tensor or type(arr) == torch.nn.Parameter
    if istorch:
        pad = [0, 0, 0, 0]
    else:
        pad = [[0, 0]] * (len(arr.shape))
    if h < hw:
        crop_height = min(h, hw)
        top = hw // 2 - crop_height // 2
        arr = arr[..., top:top + crop_height, :]
    elif h > hw:
        do_pad = True
        if istorch:
            pad[-2] = int(np.ceil((h - hw) / 2))
            pad[-1] = int(np.floor((h - hw) / 2))
        else:
            pad[-2] = [int(np.ceil((h - hw) / 2)), int(np.floor((h - hw) / 2))]
    if w < ww:
        crop_width = min(w, ww)
        left = ww // 2 - crop_width // 2
        arr = arr[..., :, left:left + crop_width]
    elif w > ww:
        do_pad = True
        if istorch:
            pad[0] = int(np.ceil((w - ww) / 2))
            pad[1] = int(np.floor((w - ww) / 2))
        else:
            pad[-1] = [int(np.ceil((w - ww) / 2)), int(np.floor((w - ww) / 2))]
    if do_pad:
        if istorch:
            arr = torch.nn.functional.pad(arr, pad, mode=mode)
        else:
            arr = np.pad(arr, pad, mode=mode)
    return arr


def save_images_as_zip(images, root_path, zip_filename="images.zip"):
    folder = os.path.join(root_path, 'images', zip_filename.split('.')[0])
    os.makedirs(folder, exist_ok=True)
    
    # Remove existing files inside the folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    # saving png files
    image_paths = []
    for i, img_array in enumerate(images):
        img = einops.rearrange(img_array, 'c h w -> h w c')
        if img.shape[-1] == 1:
            img = einops.repeat(img, 'h w 1 -> h w 3')
        img_path = os.path.join(folder, f"image_{i+1}.png")
        plt.imsave(img_path, img, vmin=0, vmax=1)
        image_paths.append(img_path)
    
    with zipfile.ZipFile(os.path.join(root_path, 'images', zip_filename), 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_path in image_paths:
            zipf.write(img_path, os.path.basename(img_path))

    
class grayscale:
    def __call__(self, x):
        R = x[..., 0:1, :, :]
        G = x[..., 1:2, :, :]
        B = x[..., 2:3, :, :]
        return 0.2126*R + 0.7152*G + 0.0722*B
    
def log_image(formatted_images, name):
    n, c, h, w = formatted_images.shape
    if c == 1:
        formatted_images = einops.repeat(formatted_images, 'n 1 h w -> n 3 h w')
    
    plt.imsave(f"./images/{name}.png", 
               einops.rearrange(np.hstack(formatted_images),'c h w -> h w c'))

class DF:
    def __init__(self):
        downscale_factor = [1,1]
        initialization_type = 'focusing_lens'
        # ============ define DFlat PSF simulator and image generator
        # 1. initialize the target phase profile of the metasurface
        self.wavelength_set_m = np.array([550e-9])  # if we are using hsi/rgb images, this should change
        out_distance_m = 1e-2
        depth_set_m = [500]
        fshift_set_m = [[0,0]]
        self.ps_locs = torch.tensor([[0,0]], dtype=torch.float32)  # point spread locations (we use a sparse grid)
        self.depth_min = 0.5
        self.depth_max = 5
        self.in_dx_m = [3e-6, 3e-6]
        self.out_dx_m = [1e-6, 1e-6]  # pixel pitch
        h, w = 512,512
        # find the MS center positions of the array
        self.array_size = [5,5]  # col, row
        self.array_spacing = [2e-3, 2e-3] # col, row
        self.MS_size = [(w+1)*self.in_dx_m[0], (h+1)*self.in_dx_m[1]]  # width, height
        self.MSArray_size = [self.array_size[0]*self.array_spacing[0] + self.MS_size[0], self.array_size[1]*self.array_spacing[1] + self.MS_size[1]] # width, height
        
        self.MS_pos = []
        for i in range(self.array_size[0]): # cols
            for j in range(self.array_size[1]): # rows
                self.MS_pos.append([(i)*self.array_spacing[0] - (self.MSArray_size[0])/2 + (self.MS_size[0]),   # x
                                    (j)*self.array_spacing[1] - (self.MSArray_size[1])/2 + (self.MS_size[1]),   # y
                                    0])                                                                   # z
        self.MS_pos = torch.tensor(self.MS_pos)

        # find the out_size of the MS. Downscale to preserve memory
        self.out_size = [h//downscale_factor[0], w//downscale_factor[1]]

        if initialization_type == 'focusing_lens':
            lenssettings = {
                "in_size": [h+1, w+1],
                "in_dx_m": self.in_dx_m,
                "wavelength_set_m": self.wavelength_set_m,
                "depth_set_m": depth_set_m,
                "fshift_set_m": fshift_set_m,
                "out_distance_m": out_distance_m,
                "aperture_radius_m": None,
                "radial_symmetry": False  # if True slice values along one radius
                }
            self.amp, self.phase, self.aperture = focusing_lens(**lenssettings) # [Lam, H, W]
            print("amp phase aperture", self.amp.shape, self.phase.shape, self.aperture.shape)
        else:
            raise NotImplementedError()

        # 2. Reverse look-up to find the metasurface that implements the target profile
        model_name = 'Nanocylinders_TiO2_U300H600'
        self.p_norm, self.p, err = reverse_lookup_optimize(
            self.amp[None, None],
            self.phase[None, None],
            self.wavelength_set_m,
            model_name,
            lr=1e-1,
            err_thresh=1e-6,
            max_iter=100,
            opt_phase_only=False)
        
        # need to move to GPU before parameter creation, or else move after optimizer creation
        def get_noise():
            return 0
            return np.random.normal(self.p.mean(), self.p.std(), np.prod(self.p.shape)).reshape(self.p.shape)
        self.p = [torch.from_numpy(self.p + get_noise()).cuda() for _ in range(len(self.MS_pos))]
        # [B, H, W, D] where D = model.dim_in - 1 is the number of shape parameters

        # 3. load optical model
        self.optical_model = load_optical_model(model_name).cuda()

        # 4. setup PSF generators from phase, amp
        # Compute the point spread function given this broadband stack of field amplitude and phases
        self.PSF = PointSpreadFunction(
            in_size=[h+1, w+1],
            in_dx_m=self.in_dx_m,
            out_distance_m=out_distance_m,
            out_size=self.out_size,
            out_dx_m=self.out_dx_m,
            out_resample_dx_m=None,
            radial_symmetry=False,
            diffraction_engine="ASM").cuda()

        # 5. renderer for image blurring
        self.renderer = Fronto_Planar_Renderer_Incoherent()
    
    def get_image(self, img, ps_locs, p):
        est_amp, est_phase = self.optical_model(p, self.wavelength_set_m, pre_normalized=False)
        psf_intensity, _ = self.PSF(
            est_amp.to(dtype=torch.float32, device=img.device),
            est_phase.to(dtype=torch.float32, device=img.device),
            self.wavelength_set_m,
            ps_locs,
            aperture=None,
            normalize_to_aperture=True)
        
        # Need to pass inputs like
        # psf has shape  [B P Z L H W]
        # scene radiance [B P Z L H W]
        # out shape      [B P Z L H W]
        meas = self.renderer(psf_intensity, einops.rearrange(img, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
        meas = self.renderer.rgb_measurement(meas, self.wavelength_set_m, gamma=True, process='demosaic')

        return meas, psf_intensity
    
    def process_data(self, fg, bg, mask, device):
        fg=fg.to(device)
        bg=bg.to(device)
        mask=mask.to(device)

        # generate the ground truth images [all-in-focus image, depth map]
        bg_depth = torch.tensor(self.depth_max) # torch.rand(1)*(self.depth_max - self.depth_min) + self.depth_min
        fg_depth = torch.tensor(self.depth_max/2) # torch.rand(1)*(bg_depth - self.depth_min) + self.depth_min
        print(f"bg {bg_depth} fg {fg_depth}")

        all_psf_intensity = {'fg':[], 'bg':[]}
        all_meas = []
        # simulate PSF for current p
        for i in range(len(self.p)):
            assert not torch.isnan(self.p[i]).sum(), f'{torch.isnan(self.p[i]).sum()} Nan in p'
            mask_meas, psf_intensity = self.get_image(
                mask, 
                torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*fg_depth], dim=-1) - self.MS_pos[i],
                self.p[i])
            fg_meas = self.renderer(psf_intensity, einops.rearrange(fg, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
            fg_meas = self.renderer.rgb_measurement(fg_meas, self.wavelength_set_m, gamma=True, process='demosaic')
            all_psf_intensity['fg'].append(psf_intensity)
            
            bg_meas, psf_intensity = self.get_image(
                bg, 
                torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*bg_depth], dim=-1) - self.MS_pos[i],
                self.p[i])
            all_psf_intensity['bg'].append(psf_intensity)
            
            # fg_meas = (fg_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
            # bg_meas = (bg_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
            # mask_meas = (mask_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
            fg_meas = fg_meas[:, 0, 0]  # no polarization
            bg_meas = bg_meas[:, 0, 0]
            mask_meas = mask_meas[:, 0, 0]
            
            # alpha clipping and merging fg and bg  
            all_meas.append(bg_meas*(1 - mask_meas) + fg_meas*mask_meas) # alpha clipping and merging fg and bg
        
        gt = bg*(1 - mask) + fg*mask # this is not correct for all MS in the array. there can be parallax effect for GT
        all_meas = torch.stack(all_meas)
        all_meas = einops.rearrange(all_meas, "n b c h w -> b n c h w")
        
        data = {}
        data['fg'] = fg
        data['bg'] = bg
        data['gt'] = gt
        data['meas'] = all_meas
        data['depth'] = bg_depth.to(device)*(1-mask) + fg_depth.to(device) *mask  # depth map
        data['depth'] = (data['depth'] - self.depth_min)/(self.depth_max - self.depth_min)
        data['mask'] = mask
        data['all_psfs'] = all_psf_intensity
        self.sample = data

    @torch.no_grad()
    def find_depth_by_shift(self, imgs, N=20, max_pix_shifts = 30):
        # imgs : nMS c h w
        shift_by = np.linspace(0, max_pix_shifts, N)
        # shift_by = (max_pix_shifts - np.log(np.linspace(1, np.exp(max_pix_shifts/10), N))*10)[::-1]
        h, w = imgs.shape[-2:]
        cols, rows = self.array_size
        original_idx = rows//2 * cols + cols//2  # center of the grid
        device = imgs.device
        assert imgs.shape[0] == rows*cols, f"requires {rows*cols} images corresponding to each MS in the array"
        
        # direction to all other MS
        dirs = [self.MS_pos[i,:2] - self.MS_pos[original_idx,:2] for i in range(len(self.MS_pos))]
        dir_norm_max = max([torch.norm(dirs[i]) for i in range(len(dirs))])
        dirs = torch.stack([dirs[i]/dir_norm_max for i in range(len(dirs))]).to(device)
        
        xx, yy = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='xy')
        pix_pos = einops.rearrange(torch.stack([xx, yy]), 't h w -> h w t').to(device).to(torch.float32)

        shifted_imgs = []
        for i in range(len(self.MS_pos)):
            # if i == original_idx: continue
            # getting shifted images
            shifted_pix_pos = []
            for n in range(N):
                idx = pix_pos + dirs[i][None, None]*shift_by[n]
                #grid sample needs from -1 to +1 
                idx[..., 0] = idx[..., 0]/w*2 - 1
                idx[..., 1] = idx[..., 1]/h*2 - 1
                shifted_pix_pos.append(idx) 
            shifted_pix_pos = torch.stack(shifted_pix_pos)
            shifted_imgs.append(torch.nn.functional.grid_sample(einops.repeat(imgs[i:i+1], '1 c h w -> n c h w', n=N), shifted_pix_pos, align_corners=False))
        shifted_imgs = torch.stack(shifted_imgs) # nMS, N, c, h, w
        # save_images_as_zip(einops.rearrange(shifted_imgs, '(n1 n2) N c h w -> N c (n2 h) (n1 w)', n1=cols, n2=rows).cpu().numpy(), '.', "shifted.zip")

        # should we do window matching?
        shifted_pix_std = torch.std(shifted_imgs - imgs[None, None, original_idx], dim=0).cpu().numpy()
        shifted_pix_std /= shifted_pix_std.max()
        shifted_pix_std = shifted_pix_std.mean(-3, keepdims=True)
        # log_image(shifted_pix_std.argmin(0)[None]/N, 'dispatity_by_shift_min')
        # log_image(shifted_pix_std.argmax(0)[None]/N, 'dispatity_by_shift_max')
        depth = 1/(shifted_pix_std.argmin(0)[None]+1) 
        # depth = depth/depth.max()
        # depth = np.clip(depth, 0, 1)
        return depth, shifted_pix_std
    
    @torch.no_grad()
    def test1(self, object_images):
        # plot meas at different depths
        depths = np.linspace(self.depth_min, self.depth_max, 10)
        # print("depth levels", depths)
        imgs = []
        psfs = []
        est_depths = [[] for _ in range(len(depths))]
        for d, depth in enumerate(depths):
            meas_arr = []
            psfs_arr = []
            for i in range(len(self.p)):
                meas, psf_intensity = self.get_image(
                    object_images,
                    torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*depth], dim=-1) - self.MS_pos[i],
                    self.p[i])
                meas_arr.append(meas[:, 0, 0])
                psfs_arr.append(psf_intensity[:, 0, 0])
            psfs_arr = torch.stack(psfs_arr)[..., :, 200:300, 200:300]  # TODO: change hard-coded croppping
            meas_arr = torch.stack(meas_arr)
            imgs.append(einops.rearrange(meas_arr, '(n1 n2) b c h w -> b c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1]))
            psfs.append(einops.rearrange(psfs_arr, '(n1 n2) b c h w -> b c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1]))
            for i in range(meas_arr.shape[1]): # iterate over batch
                depth_map, shifted_pix_std = self.find_depth_by_shift(meas_arr[:, i], N=20, max_pix_shifts=100)
                # log_image(depth_map, f'depth_by_shift_d-{depth*10}')
                # save_images_as_zip(shifted_pix_std, '.', f'depth_by_shift_std-{depth*10}.zip')
                est_depths[d].append(np.mean(depth_map))
                # print(est_depths[d][-1], depth)
        # log_image(object_images.cpu(), f'images')
        psfs = torch.cat(psfs, 0).cpu().numpy()
        psfs /= psfs.max()
        # save_images_as_zip(torch.cat(imgs, 0).cpu().numpy(), '.', f'imgs_at_depths.zip')
        # save_images_as_zip(psfs, '.', f'psfs_at_depths.zip')
        return est_depths, depths
    
    @torch.no_grad()
    def plot(self, idx):
        lq_key, gt_key = 'meas', 'gt'

        # plot PSFs
        psfs = {}
        all_psf_intensity = self.sample['all_psfs']
        for i in range(len(self.p)):
            psf_intensity = torch.cat([all_psf_intensity['fg'][i],all_psf_intensity['bg'][i]], dim=2)
            print(f'ms {i} psf',psf_intensity.shape)
            for j in range(psf_intensity.shape[2]):
                psf = psf_intensity[0,0,j].detach().cpu().numpy()
                if j in psfs.keys():
                    psfs[j].append(psf)
                else:
                    psfs[j] = [psf]
        for k in psfs.keys():
            is_fg = k < len(self.ps_locs)
            image1 = np.stack(psfs[j])
            image1 = image1[..., :, 200:300, 200:300] # TODO: change hard-coded croppping
            image1 = einops.rearrange(image1, '(n1 n2) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
            image1 = np.stack([image1])
            image1 = image1/image1.max()
            image1 = np.clip(image1, 0, 1)
            ps_loc = self.ps_locs[k%len(self.ps_locs)]
            log_image(image1, f"img{idx}_psfs_{k}_{'fg' if is_fg else 'bg'}_{ps_loc}")

        # plot blurred image captures on sensor
        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        gt_depth = self.sample['depth']
        lq = lq.cpu().numpy()
        gt = gt.cpu().numpy()
        gt_depth = gt_depth.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            save_images_as_zip(lq[i], '.', f'lq_{idx}.zip')
            lq_i = einops.rearrange(lq[i], '(n1 n2) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
            lq_i = np.stack([lq_i])
            lq_i = np.clip(lq_i, 0, 1)
            log_image(lq_i, f'lq_{idx}')

            image1 = [gt[i], gt_depth[i]]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(image1, f'{idx}')       
        
        # calculate Differential stereo and plot
        # formula: d_est = (I_R - I_L) / (dI_R/dx + epsilon)
        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        gt_depth = self.sample['depth']
        all_d_est = []
        epsilon = 1e-6
        cols, rows = self.array_size
        for i in range(rows):
            for j in range(cols):
                left_img = lq[:, i*cols + j]
                right_img = lq[:, i*cols + (j + 1)%cols]

                # Compute the horizontal gradient of the right image using a finite-difference kernel.
                kernel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=right_img.device).view(1, 1, 1, 3) / 2.0
                kernel_x = einops.repeat(kernel_x, 'b 1 h w -> b 3 h w')
                right_grad_x = torch.nn.functional.conv2d(right_img, kernel_x, padding=(0, 1))

                d_est = (right_img - left_img) / (right_grad_x + epsilon)
                d_est = torch.clip(d_est, -10, 10)
                d_est = (d_est - d_est.min())/(d_est.max()-d_est.min())
                d_est_img = d_est.detach().cpu()
                all_d_est.append(d_est_img)
        all_d_est = einops.rearrange(torch.stack(all_d_est), 'n b c h w -> b n c h w')
        d_est = einops.rearrange(all_d_est, 'b (n1 n2) c h w ->b c (n2 h) (n1 w)', n1=cols, n2=rows)
        d_est =  d_est/d_est.max()
        log_image(d_est, f'depth_by_stereo_{idx}')


def main():
    fg = plt.imread('fg.png')[:128,:128,:3]
    bg = plt.imread('bg.png')[:128,:128,:3]
    mask = plt.imread('mask.png')[:128,:128,:3]
    fg = cv2.resize(fg, (512,512))
    bg = cv2.resize(bg, (512,512))
    mask = cv2.resize(mask, (512,512))
    fg = einops.rearrange(fg, 'h w c -> 1 c h w')
    bg = einops.rearrange(bg, 'h w c -> 1 c h w')
    mask = einops.rearrange(mask, 'h w c -> 1 c h w')
    
    gray = grayscale()
    fg = gray(fg)
    bg = gray(bg)
    mask = gray(mask)

    print(fg.shape, bg.shape, mask.shape)
    fg,bg,mask = torch.from_numpy(fg), torch.from_numpy(bg), torch.from_numpy(mask)

    df = DF()
    df.process_data(fg,bg, mask, 'cuda')
    df.plot(1)
    
    # depth by comparison
    def plot_std(p, q):
        ys = shifted_pix_std[:, 0, p, q]
        xs = np.arange(len(ys))
        p = np.poly1d(coeff[:, p, q])
        plt.plot(xs,ys)
        plt.plot(xs,p(xs), '--')
        plt.title(f"{p}, {q}")
        plt.show()
    lq = df.sample['meas']
    for _i in range(len(lq)):
        depth, shifted_pix_std = df.find_depth_by_shift(lq[_i], N=20, max_pix_shifts=30)
        print(depth.shape)
        # shifted_pix_std.shape (N 1 H W)
        N, _, H, W = shifted_pix_std.shape
        D = 4
        coeff = np.polyfit(np.arange(N), einops.rearrange(shifted_pix_std[:,0], 'n h w -> n (h w)'), D)
        P_prime = polyder(coeff) # First derivative
        P_double_prime = polyder(P_prime) # Second derivative
        
        depth_fine = []
        for i in range(P_prime.shape[1]):
            roots = np.roots(P_prime[:, i])
            roots = roots[np.isreal(roots)].real # Filter out any complex solutions (keep real values only)
            minimum = 1e10
            minimum_root = 1e10
            P = np.poly1d(coeff[:, i])
            for root in roots:
                if not (root>=0 and root<=N):
                    continue
                val = P(root)
                if minimum > val:
                    minimum = val
                    minimum_root = root
            depth_fine.append(minimum_root)
        depth_fine = einops.rearrange(np.array(depth_fine), '(h w) -> 1 1 h w', h=H,w=W)
        depth_fine = 1/depth_fine
        print(depth_fine.min(), depth_fine.max())
        depth_fine = np.clip(depth_fine, 0, 1.0)
        coeff = einops.rearrange(coeff, 'd (h w) -> d h w', h=H,w=W)
        log_image(depth, 'depth_by_shift')
        log_image(depth_fine, 'depth_by_shift_fine')
        save_images_as_zip(shifted_pix_std, '.', f'depth_by_shift_std.zip')
        
        plot_std(100, 100)
        plot_std(183, 270)
            
    
    ds = load_dataset('harshana95/BackgroundDataset', split='train', trust_remote_code=True)
    import torchvision.transforms.functional as F
    
    batch = []
    est_depths = None
    i = 0
    for sample in ds:
        img = F.to_tensor(sample['background'])[:3, :, :]
        img = gray(img)
        img = crop_arr(img, 128,128)
        img = torch.nn.functional.interpolate(img[None], (128*4, 128*4), mode='bicubic')[0]
        batch.append(img)
        i += 1
        if i%16 == 0:
            batch = torch.stack(batch).to('cuda')
            batch = torch.clip(batch, 0.0, 1.0)
            # print(batch.shape, batch.min(), batch.max())
            est_depth, depths = df.test1(batch)
            if est_depths is None:
                est_depths = est_depth
            else:
                for d in range(len(est_depth)):
                    est_depths[d] += est_depth[d]
            act_d = []
            est_d = []
            for d in range(len(est_depths)):
                act_d += [depths[d]]*len(est_depths[d])
                est_d += est_depths[d]
            data = pd.DataFrame({'Actual depth': act_d, "Est depth": est_d})
            data['Est depth'] *= 12  # arbitary number
            plt.figure(figsize=(8,6))
            sns.lineplot(x='Actual depth', y='Est depth', data=data, markers='o', estimator=np.mean, errorbar='sd')
            plt.xlabel("Actual depth (m)")
            plt.ylabel("Estimated depth (m)")
            plt.title(f"{len(data)}")
            plt.savefig('./images/error.png')
            plt.close()
            batch = []
            print(np.mean(np.array(est_depths), 1))
        

if __name__ == '__main__':
    main()
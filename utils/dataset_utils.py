import PIL
import einops
import kornia
import numpy as np
import torch
from matplotlib import pyplot as plt
# from pynoise.noisemodule import Perlin
# from pynoise.noiseutil import grayscale_gradient, RenderImage, noise_map_plane, noise_map_plane_gpu
from torchvision.transforms import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as TF
from torch.fft import fft2, ifft2, fftshift, ifftshift


class DictWrapper:
    def __init__(self, f, keys, new_keys=None):
        self.f = f
        self.keys = keys
        if new_keys is None:
            new_keys = []
        self.new_keys = new_keys if len(new_keys) > 0 else keys

    def __call__(self, sample):
        for key, new_key in zip(self.keys, self.new_keys):
            sam = sample[key]
            sample[new_key] = [self.f(image) for image in sam]
        return sample
    
    def __str__(self):
        self.f
        
class identity:
    def __call__(self, x):
        return x
    
class apply_model:
    def __init__(self, model_name, model_path, **kwargs):
        from utils.misc import find_attr
        from models.archs import _arch_modules
        c = find_attr(_arch_modules, model_name)
        self.model = c.from_pretrained(model_path)
        print(f"Loaded {model_name} model from {model_path}")
        self.model.eval()
        self.model = self.model.cuda()  # always run the model in GPU
        self.kwargs = kwargs
    
    @torch.no_grad()
    def __call__(self, x):
        device = x.device
        x = x.to(self.model.device)
        # print(x.shape, x.device)
        reduce_batch_dim = False
        if len(x.shape) == 3:
            x = x[None]
            reduce_batch_dim = True
        ret = self.model(x, **self.kwargs)
        if reduce_batch_dim:
            ret = ret[0]
        return ret.to(device)
    
class apply_function:
    def __init__(self, func, **kwargs):
        import utils.image_utils as image_utils
        self.func = getattr(image_utils, func, None)
        self.kwargs = kwargs
    def __call__(self, x):
        return self.func(x, **self.kwargs)
    
class sv_convolution:
    def __init__(self, basis_psfs, basis_coef, **kwargs):
        self.basis_psfs = einops.rearrange(basis_psfs, 'c 1 n h w -> c n h w')
        self.basis_coef = einops.rearrange(basis_coef, 'c 1 n h w -> c n h w')
        assert self.basis_coef.shape[-2:] == self.basis_psfs.shape[-2:], f"Shape mismatch {self.basis_coef.shape} {self.basis_psfs.shape}"

        self.H = fft2(self.basis_psfs, dim=(-2, -1))
        self.W = self.basis_coef
    
    def __call__(self, x, *args, **kwds):
        # x: (c h w)
        # W: (c n h w)
        # H: (c n h w)
        X = fft2(x[:, None] * self.W, dim=(-2, -1))
        out = ifftshift(ifft2(X * self.H, dim=(-2, -1)), dim=(-2, -1)).real.sum(-3)
        return out
    
class homography:
    def __init__(self, M):
        self.M = torch.Tensor(M)[None]
        print("homography matrix", self.M)
    def __call__(self, x):
        self.M = self.M.to(x.device)
        return kornia.geometry.warp_perspective(x[None], self.M, dsize=(x.shape[-2], x.shape[-1]))[0]

class to_tensor:
    # BGR to RGB, HWC to CHW, numpy to tensor
    def __init__(self):
        self.f = transforms.ToTensor()
    def __call__(self, x):
        return self.f(x)
    
class grayscale:
    def __call__(self, x):
        if x.shape[-3] == 1:
            return x
        R = x[..., 0:1, :, :]
        G = x[..., 1:2, :, :]
        B = x[..., 2:3, :, :]
        return 0.2126*R + 0.7152*G + 0.0722*B
    

class rescale:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        squeeze = False
        if len(x.shape) == 3:
            x = x[None]
            squeeze = True
        assert len(x.shape) == 4, x.shape
        h, w = x.shape[-2] * self.factor, x.shape[-1] * self.factor
        arr = torch.nn.functional.interpolate(x, (int(h), int(w)), mode='bicubic')
        return arr[0] if squeeze else arr
    
class resize:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x):
        squeeze = False
        if len(x.shape) == 3:
            x = x[None]
            squeeze = True
        assert len(x.shape) == 4, x.shape
        h, w = self.h, self.w
        arr = torch.nn.functional.interpolate(x, (int(h), int(w)), mode='bicubic')
        return arr[0] if squeeze else arr
    
class crop:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x):
        return crop_arr(x, self.h, self.w)
    
class crop_idx:
    def __init__(self, h1, h2, w1, w2):
        self.h1 = h1
        self.h2 = h2
        self.w1 = w1
        self.w2 = w2

    def __call__(self, x):
        x = x[..., self.h1:self.h2, self.w1:self.w2]
        return x

class crop_random:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def crop(self, image):
        h, w = image.shape[-2], image.shape[-1]
        if h <= self.h and w <= self.w:
            return image
        top = np.random.randint(0, h - self.h)
        left = np.random.randint(0, w - self.w)
        return image[..., top:top+self.h, left:left+self.w]

    def __call__(self, sample):
        return  self.crop(sample)
    

class select_channels:
    def __init__(self, channels):
        if isinstance(channels, list) or isinstance(channels, tuple):
            channels = torch.Tensor(channels)
        elif isinstance(channels, np.ndarray):
            channels = torch.from_numpy(channels)
        self.channels = channels.to(torch.bool)

    def __call__(self, x):
        if x.shape[-3] == len(self.channels):
            return x[..., self.channels, :, :]
        return x

class gaussian_noise:
    def __init__(self, max_sigma=0.02, sigma_dc=0.005, mu=0):
        self.max_sigma = max_sigma  # abit much maybe 0.04 best0.04+0.01
        self.sigma_dc = sigma_dc
        self.mu = mu

    def add_noise(self, sample):
        shape = sample.shape
        sigma = np.random.rand() * self.max_sigma + self.sigma_dc
        g_noise = torch.empty(shape).normal_(mean=self.mu, std=sigma).to(sample.dtype).to(sample.device)
        ret = sample + g_noise
        ret = torch.clamp(ret, 0, 1)
        return ret#, torch.tensor([sigma])

    def __call__(self, sample):
        return self.add_noise(sample)


class poisson_noise:
    def __init__(self,peak=1000, peak_dc=50):
        super().__init__()
        self.PEAK = peak  # np.random.rand(1) * 1000 + 50
        self.PEAK_DC = peak_dc
        
    def add_noise(self, sample):
        peak = np.random.rand() * self.PEAK + self.PEAK_DC
        if peak < 0:
            return sample, torch.tensor([0])
        p_noise = torch.poisson(torch.clamp(sample, min=1e-6) * peak)  # poisson cannot take negative
        p_noise = p_noise.to(sample.device)
        # ret = p_noise
        ret = (p_noise.to(sample.dtype) / peak)  # poisson noise is not additive
        # ret = ret / torch.max(ret)
        # ret = torch.maximum(ret, torch.zeros_like(ret))
        ret = torch.clamp(ret, 0, 1)
        return ret#, torch.tensor([peak])

    def __call__(self, sample):
        return self.add_noise(sample)


# class perlin_noise:
#     def __init__(self, refresh_noise_for_each=True):
#         super().__init__()
#         self.refresh_noise_for_each = refresh_noise_for_each
#         self.p = Perlin(frequency=6, octaves=10, persistence=0.6, lacunarity=2, seed=0)
#         self.gradient = grayscale_gradient()
#         self.render = RenderImage(light_enabled=True, light_contrast=3, light_brightness=2)
#         self.lx, self.ux = 100, 200
#         self.lz, self.uz = 100, 200
#         self.noise_min = 0.8
#         self.noise_max = 1.0

#         self.nm = None
#         self.tmp_h, self.tmp_w = -1, -1

#     def add_noise(self, image):
#         h, w = image.shape[-2:]
#         if self.nm is None or h != self.tmp_h or w != self.tmp_w or self.refresh_noise_for_each:
#             self.p = Perlin(frequency=6, octaves=10, persistence=0.6, lacunarity=2, seed=np.random.randint(1e6))
#             self.nm = noise_map_plane_gpu(width=w, height=h,
#                                           lower_x=self.lx, upper_x=self.ux,
#                                           lower_z=self.lz, upper_z=self.uz,
#                                           source=self.p)
#             self.tmp_h = h
#             self.tmp_w = w
#         noise: PIL.Image = self.render.render(w, h, self.nm, 'remove_img_save_from_source.png', self.gradient)
#         if noise is None:
#             raise Exception("!!!!!!!!!! Return image in noiseutil.py in pynoise package !!!!!")
#         noise = pil_to_tensor(noise)
#         if image.max() <= 1.0:
#             noise = noise.to(torch.float32).to(image.device)
#             noise /= 255
#             noisy = image * ((noise*(self.noise_max - self.noise_min)) + self.noise_min)
#         else:
#             noise = noise * (int(255 * self.noise_max) - int(255*self.noise_min)) + int(255*self.noise_min)
#             noisy = image * noise // 255
#             # noisy[noisy < noise//2] = 255  # avoid overflowing
#         return noisy

#     def __call__(self, sample):
#         return self.add_noise(sample)


class padding:
    def __init__(self, h, w, mode='reflect'):
        super().__init__()
        self.h = h
        self.w = w
        self.mode = mode

    def __call__(self, sample):
        return crop_arr(sample, self.h, self.w, mode=self.mode)
        

class normalize:
    def __init__(self, mean, std, inplace=False):
        self.norm = transforms.Normalize( mean, std, inplace)  # (x - mean) / std

    def __call__(self, sample):
        return self.norm(sample)


class rotate:
    def __init__(self, angle):
        self.ops = []
        self.angle = angle

    def rotate(self, image):
        return TF.rotate(image, self.angle)

    def __call__(self, sample):
        return self.rotate(sample)


class augment:
    def __init__(self, image_shape, horizontal_flip=True, resize_crop=True):
        self.ops = []
        if horizontal_flip:
            self.ops.append(transforms.RandomHorizontalFlip())
        if resize_crop:
            self.ops.append(transforms.RandomResizedCrop(image_shape, antialias=True))

    def aug(self, image):
        for op in self.ops:
            image = op(image)
        return image

    def __call__(self, sample):
        return self.aug(sample)


class translate_image:
    def __init__(self, translate_by):
        self.translate_by = translate_by

    def translate(self, image, idx):
        translate = self.translate_by[idx % len(self.translate_by)]
        translated = torch.zeros_like(image)
        c, h, w = image.shape
        hs, he, ws, we = max(0, translate[0]), min(h, translate[0] + h), max(0, translate[1]), min(w, translate[1] + w)
        ihs = abs(min(0, translate[0]))
        iws = abs(min(0, translate[1]))
        ihe = ihs + (he - hs)
        iwe = iws + (we - ws)
        translated[:, hs:he, ws:we] = image[:, ihs:ihe, iws:iwe]
        return translated

    def __call__(self, sample, astype=torch.float32):
        
        return self.translate(sample, sample['idx']).to(astype)
        


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
            print(pad)
            arr = np.pad(arr, pad, mode=mode)
    return arr


def bartlett(ph=32, pw=32, color=True):
    x = np.arange(1, pw + 1, 1.0) / pw
    y = np.arange(1, ph + 1, 1.0) / ph

    xx, yy = np.meshgrid(x, y)
    a0 = 0.62
    a1 = 0.48
    a2 = 0.38
    win = a0 - a1 * np.abs(xx - 0.5) - a2 * np.cos(2 * np.pi * xx)
    win *= a0 - a1 * np.abs(yy - 0.5) - a2 * np.cos(2 * np.pi * yy)
    win = win[None, :, :]
    if color:
        win = np.repeat(win, 3, 0)
    return win


def merge_patches(patches, pos):
    """

    @param patches: patches with shape (n, c, ph, pw)
    @param pos: patch top-left position (n, 2)
    @return:
    """
    n, c, ph, pw = patches.shape
    window = torch.tensor(bartlett(ph, pw, color=c == 3), device=patches.device)

    # find the image size
    h, w = pos[:, 0].max() + ph, pos[:, 1].max() + pw
    out = torch.zeros((c, h, w), device=patches.device)
    out_weights = torch.zeros_like(out, device=patches.device)
    for patch, p in zip(patches, pos):
        i, j = p
        out[:, i:i + ph, j:j + pw] += patch * window
        out_weights[:, i:i + ph, j:j + pw] += window
    out /= (out_weights + 1e-6)
    return out


def patchify(arr, ph, pw, sh, sw):
    """
    Patchify an array
        arr: input image Tensor array (b c h w)
        ph: patch height
        pw: patch width
        sh: stride height
        sw: stride width
        return: patch array (b n c ph pw), patch position
    """
    size_w = arr.shape[-1]
    exp_size_w = int(np.ceil((size_w - pw) / sw)) * sw + pw
    exp_size_w = exp_size_w + pw if exp_size_w < size_w else exp_size_w

    size_h = arr.shape[-2]
    exp_size_h = int(np.ceil((size_h - ph) / sh)) * sh + ph
    exp_size_h = exp_size_h + ph if exp_size_h < size_h else exp_size_h

    if exp_size_w > size_w or exp_size_h > size_h:
        arr = torch.nn.functional.pad(arr, (0, exp_size_w - size_w, 0, exp_size_h - size_h))

    # patch arr
    patched = arr.unfold(-2, ph, sh).unfold(-2, pw, sw)
    patched = einops.rearrange(patched, "... c n1 n2 ph pw -> ... (n1 n2) c ph pw")
    patched_pos = torch.tensor(np.mgrid[0:exp_size_h - ph + 1:sh,
                               0:exp_size_w - pw + 1:sw].reshape(2, -1).T)
    return patched, patched_pos


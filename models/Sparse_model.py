import cv2
import einops
import numpy as np
import torch
from tqdm import tqdm
import gc
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from models.archs import define_network
from models.base_model import BaseModel
from utils.dataset_utils import grayscale, merge_patches, poisson_noise
from utils.loss import Loss
from utils import log_image, log_metrics

def get_edges(img, z):  # todo : send the depth map instead of z 
    # optical parameters, unit is meter
    zs = 0.01 
    pp = 1e-6
    Sigma = 1e-3
    dSigma = 3e-4
    zf = 2
    rho = 1/zf + 1/zs
    lower_threshold = 4/255
    upper_threshold = 8/255
    # alpha = 50000
    drho = 0.00
    drho_VP = 0.21
    poisson = poisson_noise(peak=50000, peak_dc=50)
    gray = grayscale()
    # def add_poisson_noise(image, alpha=1):
    #     # Ensure the image is a float type
    #     image = image.astype(np.float64)
    #     #Determine max value for scaling
    #     scaled_image = image * alpha / 255
    #     # Generate Poisson noise with the scaled image as the mean
    #     noisy_image = np.random.poisson(scaled_image)
    #     # Scale back to the original range and convert to the original data type
    #     noisy_image = np.minimum(noisy_image / alpha * 255, 255).astype(np.uint8)
    #     return noisy_image

    def psf(z, rho, Sigma, device):
        sigma = ((1/z - rho) * zs + 1) / pp * Sigma
        sigma = np.abs(sigma)
        # print(z, rho, Sigma, sigma)
        if sigma < 0.1:
            return torch.tensor([[1.0]], dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(torch.arange(-3*sigma, 3*sigma+1e-6, device=device), torch.arange(-3*sigma, 3*sigma+1e-6, device=device))
        return torch.exp(-(xx**2 + yy**2) / (2 * sigma**2)) / (2 * torch.pi * sigma**2)


    def psf_set(z, device):
        R_HP_psf = psf(z, rho - drho, Sigma, device)
        G_HP_psf = psf(z, rho, Sigma, device)
        B_HP_psf = psf(z, rho + drho, Sigma, device)

        R_VP_psf = psf(z, rho - drho + drho_VP, Sigma, device)
        G_VP_psf = psf(z, rho + drho_VP, Sigma, device)
        B_VP_psf = psf(z, rho + drho + drho_VP, Sigma, device)

        R_DP_psf = psf(z, rho - drho, Sigma + dSigma, device)
        G_DP_psf = psf(z, rho, Sigma + dSigma, device)
        B_DP_psf = psf(z, rho + drho, Sigma + dSigma, device)

        return [R_HP_psf, G_HP_psf, B_HP_psf, R_VP_psf, G_VP_psf, B_VP_psf, R_DP_psf, G_DP_psf, B_DP_psf]
    
    def render_img(img, z, mode='color'):
        # print('Rendering image at z =', z)
        psf_set_img = psf_set(z, img.device)  # todo get psf set from depthmap
        img_HP = torch.zeros_like(img) # horizontal linear polarization
        img_VP = torch.zeros_like(img) # vertical linear polarization
        img_DP = torch.zeros_like(img) # diagonal linear polarization
        for i in range(3):
            img_HP[:,i:i+1,:,:] = poisson(F.conv2d(img[:,i:i+1,:,:],  psf_set_img[i][None, None], padding='same'))
            img_VP[:,i:i+1,:,:] = poisson(F.conv2d(img[:,i:i+1,:,:],  psf_set_img[i+3][None, None], padding='same'))
            img_DP[:,i:i+1,:,:] = poisson(F.conv2d(img[:,i:i+1,:,:],  psf_set_img[i+6][None, None], padding='same'))
            
        if mode == 'gray':
            return gray(img_VP), gray(img_HP), gray(img_DP)
        
        return img_VP, img_HP, img_DP

    def image_subtraction(img1, img2):
        return (img1 - img2)

    def keep_positive(img):
        img[img<0] = 0
        return img

    def omit_small_values(img, lower_threshold, upper_threshold):
        img[torch.abs(img) <= lower_threshold] = 0 
        # img[img >= upper_threshold] = upper_threshold
        # img[img <= -upper_threshold] = -upper_threshold
        img[img >= upper_threshold] = 0
        img[img <= -upper_threshold] = 0
        return img
    
    def binarize_img(img, z):
        img_VP, img_HP, img_DP = render_img(img, z, mode='gray')
        img_processed = image_subtraction(img_HP, img_VP)
        img_processed = keep_positive(img_processed)
        img_processed = omit_small_values(img_processed, lower_threshold, upper_threshold)
        img_processed[img_processed > 0] = 1
        return img_processed

    return binarize_img(img, z)

def sparse_color(img, sparsity=0.001):
    mask = torch.rand_like(img[:, 0:1]) < sparsity
    return img*mask

def pixelate_sparse_color(img_sparse):
    kernel = np.ones((11,1))
    img_sparse = cv2.filter2D(img_sparse, -1, kernel)
    img_sparse = cv2.filter2D(img_sparse, -1, kernel.T)
    return img_sparse

class Sparse_model(BaseModel):

    def __init__(self, opt, logger):
        super(Sparse_model, self).__init__(opt, logger)

        # define network
        self.net_g = define_network(opt.network)
        self.models.append(self.net_g)
        print(self.net_g)
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)    

    def feed_data(self, data, is_train=True):
        data['edges'] = get_edges(data['image'], torch.rand(1).item()*1.5 +  0.5)  # todo send depth map instead of random depth
        data['sparse_image'] = sparse_color(data['image'])
        data['sparse_depth'] = sparse_color(data['depth'])
        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=["edges", "sparse_image", "sparse_depth"], opt=self.opt.train if is_train else self.opt.val)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        net_in = torch.cat([self.sample['edges'],self.sample['sparse_image'],self.sample['sparse_depth']], dim=1)
        preds = self.net_g(net_in)
        losses = self.criterion(preds, torch.cat([self.sample['image'],self.sample['depth']],dim=1))
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        if self.opt.val.patched:
            b,c,h,w = self.original_size['image']
            pred = []
            for _ in self.setup_patches(): 
                net_in = torch.cat([self.sample['edges'],self.sample['sparse_image'],self.sample['sparse_depth']], dim=1)
                out = self.net_g(net_in)
                pred.append(out)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample['image_patched_pos'])
                out.append(merged[..., :h, :w])
            image = self.sample['image_original']
            depth = self.sample['depth_original']
            out = torch.stack(out)
        else: 
            edges = self.sample['edges']
            image = self.sample['image']
            depth = self.sample['depth']
            sparse_image = self.sample['sparse_image']
            sparse_depth = self.sample['sparse_depth']
            net_in = torch.cat([self.sample['edges'],self.sample['sparse_image'],self.sample['sparse_depth']], dim=1)
            out = self.net_g(net_in)

        edges = edges.cpu().numpy()
        image = image.cpu().numpy()
        depth = depth.cpu().numpy()
        sparse_image = sparse_image.cpu().numpy()
        sparse_depth = sparse_depth.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(image)):
            idx += 1
            image1 = [sparse_image[i], image[i], sparse_depth[i], depth[i], edges[i], out[i, :3], out[i, 3:], np.zeros_like(image[i])]
            for j in range(len(image1)):
                if image1[j].shape[-3] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            image1 = einops.rearrange(image1, '(n1 n2) c h w -> 1 c (n1 h) (n2 w)', n1=2)
            log_image(self.opt, self.accelerator, image1, f'{idx}', self.global_step)  # image format (N,C,H,W)
            log_metrics(out[i, :3], image[i], self.opt.val.metrics, self.accelerator, self.global_step, 'image')
            log_metrics(out[i, 3:], depth[i], self.opt.val.metrics, self.accelerator, self.global_step, 'depth')
        return idx
    
    
        
import importlib
import os
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from dataset import create_dataset
from models.DFlatModels.DFlatArrayDepthONN_model import DFlatArrayDepthONN_model
from models.DFlatModels.DFlatArrayDepth_model import DFlatArrayDepth_model
from models.archs import define_network
from models.archs.related.AdaBinsMonoDepth.unet_adaptive_bins import UnetAdaptiveBins
from models.archs.related.MonoDepth2.monodepth2_arch import DepthDecoder, ResnetEncoder
from utils.dataset_utils import crop_arr, grayscale, merge_patches
from utils.image_utils import save_images_as_zip
from utils.loss import Loss
from utils import log_image, log_metrics
from utils.misc import find_attr, scandir

from models.archs import _arch_modules
def depth_metrics_torch(pred, gt, mask=None, eps=1e-6):
    """
    PyTorch version for depth metrics.
    """
    pred = pred.float()
    gt = gt.float()

    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    pred = torch.clamp(pred, min=eps)
    gt = torch.clamp(gt, min=eps)

    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    sq_rel  = torch.mean(((pred - gt) ** 2) / gt)
    rmse    = torch.sqrt(torch.mean((pred - gt) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(gt)) ** 2))

    thresh = torch.max(pred / gt, gt / pred)
    delta1 = torch.mean((thresh < 1.25).float())
    delta2 = torch.mean((thresh < 1.25 ** 2).float())
    delta3 = torch.mean((thresh < 1.25 ** 3).float())

    return {
        "AbsRel": abs_rel.item(),
        "SqRel": sq_rel.item(),
        "RMSE": rmse.item(),
        "RMSE_log": rmse_log.item(),
        "δ<1.25": delta1.item(),
        "δ<1.25²": delta2.item(),
        "δ<1.25³": delta3.item(),
    }

# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir):
    i = len(weights) - 1
    saved = {}
    while len(weights) > 0:
        weights.pop()
        model = models[i]

        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        torch.save(model.state_dict(), os.path.join(output_dir, f"{class_name}_{saved[class_name]}"))

        i -= 1

def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")
        try:
            c = find_attr(_arch_modules, class_name)
            assert c is not None
        except ValueError as e:  # class is not written by us. Try to load from diffusers
            print(f"Class {class_name} not found in archs. Trying to load from diffusers...")
            m = importlib.import_module('diffusers') # load the module, will raise ImportError if module cannot be loaded
            c = getattr(m, class_name)  # get the class, will raise AttributeError if class cannot be found    
        
        # load diffusers style into model
        model.load_state_dict(torch.load(os.path.join(input_dir, f"{class_name}_{saved[class_name]}")))
        del load_model

class DFlatArrayDepthONN_E2E_Scratch_model(DFlatArrayDepthONN_model):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        self.save_handle.remove()
        self.load_handle.remove()
        self.save_handle = self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)

        self.kernel_rescale_factor = 1
        depth_network_opt = opt.get('depth_network', None)

        if depth_network_opt.type == "monodepth2":
            # 1. Load pretrained ResNet encoder from monodepth2
            loaded_dict_enc = torch.load(depth_network_opt.encoder_path, map_location='cpu')
            loaded_dict_dec = torch.load(depth_network_opt.decoder_path, map_location='cpu')
            self.feed_height = loaded_dict_enc['height']
            self.feed_width = loaded_dict_enc['width']

            encoder = ResnetEncoder(18, False)  # ResNet-18
            # remove irrelevant keys
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
            encoder.load_state_dict(filtered_dict_enc)
            encoder.eval()
            
            # Load decoder
            depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
            depth_decoder.load_state_dict(loaded_dict_dec)
            depth_decoder.eval()
            
            encoder.requires_grad_(True)
            depth_decoder.requires_grad_(True)

            self.encoder = encoder.to(self.accelerator.device)
            self.decoder = depth_decoder.to(self.accelerator.device)

            print(f"Feed size {self.feed_height}x{self.feed_width}")

            self.models.append(self.encoder)
            self.models.append(self.decoder)

    def feed_data(self, data, is_train=True):
        assert not self.opt.train.patched
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        
        x = data[gt_key]
        d = data[lq_key]
        b, c, h, w = x.shape
        device = self.accelerator.device

        FIXED_DEPTH = 10 # in meters. We can use depth map instead but fixed now for testing
        self.all_psf_intensity = []

        # simulate PSF for current p
        for i in range(len(self.p_norm)): # iterate over all metasurfaces
            ps_locs = torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*FIXED_DEPTH], dim=-1)  # add depth to point spread locations
            ps_locs -= self.MS_pos[i] # MS at center, translate obj

            nan_check_sum = torch.isnan(self.p_norm[i]).sum()
            assert not nan_check_sum, f'{nan_check_sum} Nan in p_norm'

            est_amp, est_phase = self.optical_model(self.p_norm[i], [self.wavelength_set_m[i%len(self.wavelength_set_m)]], pre_normalized=True)
            psf_intensity, _ = self.PSF(
                est_amp.to(dtype=torch.float32, device=device),
                est_phase.to(dtype=torch.float32, device=device),
                [self.wavelength_set_m[i%len(self.wavelength_set_m)]],
                ps_locs,
                aperture=None,
                normalize_to_aperture=True)

            self.all_psf_intensity.append(psf_intensity)

        self.sample = data
        if self.opt.train.patched:
            raise Exception()
        
    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        optim_params = []
        for model in self.models:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print(f"Parameter {k} is not optimized.")
        
        # add shape parameters as a parameter to optimize
        params_to_optimize = [{'params': optim_params}, 
                              {'params': self.p_norm, 'lr': opt.learning_rate}]
        optimizer = optimizer_class(
            params_to_optimize,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(optimizer)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        all_psfs = self.all_psf_intensity

        image = self.sample[gt_key]
        depth = self.sample[lq_key]

        # predict using ms weights
        all_psfs = self.all_psf_intensity
        all_psfs = einops.rearrange(torch.stack(all_psfs), '(pn N c) 1 1 1 1 h w -> pn N c h w', pn=2, N=64, c=3)  # hard coded
        kernels = crop_arr(all_psfs, 7, 7)  # hard coded
        kernels = kernels[0] - kernels[1]  # kernels are soo small, the gradients barely update
        kernels *= self.kernel_rescale_factor

        # todo denormalize kernels

        # predict using updated kernels
        onn_pred = self.decoder(self.encoder(image, kernels))[("disp", 0)]
        
        # calculate loss
        total_loss = F.mse_loss(depth, onn_pred)*1e6

        self.accelerator.backward(total_loss)

        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return {"all": total_loss}

    def validate_step(self, batch, idx, lq_key, gt_key):
        if idx == 0:
            self.feed_data(batch, is_train=False)
            psfs = {}
            NORMALIZE_PSFS = True
            kh, kw = 7, 7
            for i in range(len(self.all_psf_intensity)):
                psf_intensity = self.all_psf_intensity[i]
                for j in range(psf_intensity.shape[2]):
                    psf = psf_intensity[0,0,j].detach().cpu().numpy()
                    if NORMALIZE_PSFS:
                        psf = psf/psf.max()
                    if i < 5:
                        log_image(self.opt, self.accelerator, psf[None], f"psf{i}_ch{j}", self.global_step)
                    psf = crop_arr(psf, kh, kw)
                    if j in psfs.keys():
                        psfs[j].append(psf)
                    else:
                        psfs[j] = [psf]
            for k in psfs.keys():
                psfs_k = np.stack(psfs[k])
                if NORMALIZE_PSFS:
                    psfs_k = psfs_k/psfs_k.sum(0, keepdims=True)
                image1 = einops.rearrange(psfs_k, '(n1 n2) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
                image1 = np.stack([image1])
                image1 = image1/image1.max()
                image1 = np.clip(image1, 0, 1)
                ps_loc = self.ps_locs[k%len(self.ps_locs)]
                log_image(self.opt, self.accelerator, image1, f"img{idx:04d}_psfs_ch{k}_{ps_loc}", self.global_step)
        
        image = self.sample[gt_key]
        depth = self.sample[lq_key]
        
        # predict using ms weights
        all_psfs = self.all_psf_intensity
        all_psfs = einops.rearrange(torch.stack(all_psfs), '(pn N c) 1 1 1 1 h w -> pn N c h w', pn=2, N=64, c=3)  # hard coded
        kernels = crop_arr(all_psfs, 7, 7)  # hard coded
        kernels = kernels[0] - kernels[1]  # out_c in_c 7 7
        kernels *= self.kernel_rescale_factor

        # predict using updated kernels
        onn_pred = self.decoder(self.encoder(image, kernels))[("disp", 0)]
        
        onn_pred = onn_pred.cpu().numpy()
        depth = depth.cpu().numpy()
        image = image.cpu().numpy()
        
        for i in range(len(onn_pred)):
            idx += 1
            # model_pred_i = np.clip(np.stack([model_pred[i]]), 0, 1)
            # log_image(self.opt, self.accelerator, model_pred_i, f'model_pred_{idx:04d}', self.global_step)

            # onn_pred_i = np.clip(np.stack([onn_pred[i]]), 0, 1)
            # log_image(self.opt, self.accelerator, onn_pred_i, f'onn_pred_{idx:04d}', self.global_step)
            
            # depth_i = np.clip(np.stack([depth[i]]), 0, 1)
            # log_image(self.opt, self.accelerator, depth_i, f'depth_{idx:04d}', self.global_step)
            
            image1 = [image[i], onn_pred[i], depth[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> c h w', c=3)
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)
            
            log_metrics(depth[i], onn_pred[i], self.opt.val.metrics, self.accelerator, self.global_step)
            ret = depth_metrics_torch(depth[i], onn_pred[i])
            for tracker in self.accelerator.trackers:
                if tracker.name == "comet_ml":
                    tracker.writer.log_metrics(ret, step=self.global_step)

        return idx


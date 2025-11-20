from matplotlib import pyplot as plt
import torch
import numpy as np
import einops
import os

from models.PSFlatent.PSF_model import PSF_model
from utils import log_image
from utils.misc import log_metric


class PSF_Emb_model(PSF_model):
    def __init__(self, opt, logger):
        super(PSF_Emb_model, self).__init__(opt, logger)

    def feed_data(self, data, is_train=True):
        lens = self.lens
        spp = self.spp

        num_points = len(data['x'])
        assert num_points % 2 == 0
        with torch.no_grad():
            # In each iteration, sample only one f_d
            # Sample (x, y), uniform distribution
            # Sample (z), Gaussian distribution (3-sigma interval)
            if is_train:
                foc_z = float(np.random.choice(self.foc_z_arr))
                x = (torch.rand(num_points) - 0.5) * 2
                y = (torch.rand(num_points) - 0.5) * 2
                z_gauss = torch.clamp(torch.randn(num_points//2), min=-3, max=3)
                x1, y1 = x[::2], y[::2]
                x2, y2 = x[1::2], y[1::2]
            else:
                batch_idx = data['x'][0]//num_points
                total_batches = len(self.test_dataloader)
                foc_z = float(self.foc_z_arr[int(len(self.foc_z_arr)*batch_idx/total_batches)])
                y, x = torch.meshgrid(torch.linspace(-1, 1, 6), torch.linspace(1, -1, 6), indexing='xy')
                x = x.flatten()
                y = y.flatten()
                if len(x) % 2 == 1:
                    x = x[:-1]
                    y = y[:-1]
                z_gauss = torch.clamp(torch.linspace(-1, 1, len(x)//2), min=-3, max=3)
                x1, y1 = x[::2], y[::2]
                x2, y2 = x[1::2], y[1::2]
            
            # refocus; changes the sensor position
            foc_dist = self.z2depth(foc_z) # mm
            lens.refocus(foc_dist)

            z = torch.zeros_like(z_gauss)
            # sample [foc_z, 1], then scale to [foc_d, dmax]
            z[z_gauss > 0] = (1 - foc_z) * z_gauss[z_gauss > 0] / 3 + foc_z
            # sample [0, foc_z], then scale to [dmin, foc_d]
            z[z_gauss < 0] = foc_z * z_gauss[z_gauss < 0] / 3 + foc_z

            # Network input, shape of [N, 4] 
            # (x,y norm to -1,1; z norm to 0,1; foc_z norm to 0,1)
            foc_z_tensor = torch.full_like(x1, foc_z)
            inp1 = torch.stack((x1, y1, z, foc_z_tensor), dim=-1)
            inp2 = torch.stack((x2, y2, z, foc_z_tensor), dim=-1)
            inp3 = (inp1 + inp2)/2

            inp = torch.cat([inp1, inp2, inp3], dim=0)
            # Ray tracing to compute PSFs, shape of [N, 3, ks, ks]
            points = torch.stack((inp[:, 0], 
                                  inp[:, 1], 
                                  self.z2depth(inp[:, 2])), dim=-1)
            psf = lens.psf_rgb(points=points, ks=self.kernel_size, spp=spp)

        
        self.sample = {
            'inp': inp.to(self.accelerator.device),
            'psf': psf.to(self.accelerator.device),
        }

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        losses = {}
        inp = self.sample['inp']
        psf = self.sample['psf']
        psf_max = psf.max()

        batch_size = len(inp)
        N = batch_size // 3

        noise_inp = torch.randn_like(psf)
        emb_inp = self.sample['inp']
        pred_psf, latent_psf = self.net_g(noise_inp, emb_inp)
        pred_psf /= self.psf_rescale_factor
        
        loss_psf  = self.criterion(pred_psf[:2*N]/psf_max, psf[:2*N]/psf_max) 
        loss_psf3 = self.criterion(pred_psf[2*N:]/psf_max, psf[2*N:]/psf_max) 

        # interpolation of latents
        latent_psf1, latent_psf2 = latent_psf[0:N], latent_psf[N:2*N]
        latent_psf3 = (latent_psf1 + latent_psf2)/2
        loss_latent = self.criterion(latent_psf3, latent_psf[2*N:])

        loss = loss_psf['all']*1e3 + loss_psf3['all']*1e3 + loss_latent['all']*1e5

        # rescale loss because the loss is too small??
        self.accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        # log_metric(self.accelerator, {'parameter mean': next(self.net_g.parameters()).mean().item()}, self.global_step)
        # breakpoint()
        for optimizer in self.optimizers:
            optimizer.step()
        return {'all': loss,
                'psf': loss_psf['all'],
                'psf3': loss_psf3['all'],
                'latent': loss_latent['all']
                }

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        psf = self.sample['psf']
        coords = self.sample['inp']
        batch_size = len(coords)
        N = batch_size // 3

        noise_inp = torch.randn_like(psf[:2*N])
        emb_inp = coords[:2*N]
        pred_psf, latents = self.net_g(noise_inp, emb_inp)
        pred_psf = pred_psf.cpu().numpy()/self.psf_rescale_factor
        real_psf = psf[:2*N].cpu().numpy()

        n = int(pred_psf.shape[0]**0.5)
        pred_psf = einops.rearrange(pred_psf, '(W H) c h w -> 1 c (H h) (W w)', H=n, W=n)
        real_psf = einops.rearrange(real_psf, '(W H) c h w -> 1 c (H h) (W w)', H=n, W=n)

        # breakpoint()
        # image1 = np.concatenate([pred_psf, pred_psf/pred_psf.max()*psf.max(), psf, abs(pred_psf-psf)], axis=-1)
        norm = real_psf.max()
        log_image(self.opt, self.accelerator, 1-np.clip(pred_psf/norm, 0, 1), f"pred_psfs", self.global_step)
        log_image(self.opt, self.accelerator, 1-np.clip(real_psf/norm, 0, 1), f"real_psfs", self.global_step)

        # check interpolated output
        # breakpoint()
        latents_inter = (latents[:N] + latents[N:2*N])/2
        latents_inter = self.net_g.sample(latents_inter)
        emb_inter_inp = self.net_g.get_embedding(coords[2*N:], latents_inter.dtype)
        pred_psf = self.net_g.decode(latents_inter, emb_inter_inp).cpu().numpy()
        real_psf = psf[2*N:].cpu().numpy()
        
        pred_psf = einops.rearrange(pred_psf, '(W H) c h w -> 1 c (H h) (W w)', H=3, W=6)
        real_psf = einops.rearrange(real_psf, '(W H) c h w -> 1 c (H h) (W w)', H=3, W=6)
        norm = real_psf.max()
        log_image(self.opt, self.accelerator, 1-np.clip(pred_psf/norm, 0, 1), f"pred_inter_psfs", self.global_step)
        log_image(self.opt, self.accelerator, 1-np.clip(real_psf/norm, 0, 1), f"real_inter_psfs", self.global_step)
        
        if idx == 0:
            os.makedirs(os.path.join(self.opt.path.experiments_root, 'images'), exist_ok=True)
            save_path = os.path.join(self.opt.path.experiments_root, 'images', f"Lens.png")
            self.lens.draw_layout(filename=save_path)
            log_image(self.opt, self.accelerator, plt.imread(save_path).transpose([2,0,1])[None], f"lens", self.global_step)

        return idx+1


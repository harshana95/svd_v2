# ------------------------------------------------------------------------
# Blur model: trains a network to generate blurred image from GT (sharp) image + metadata.
# Input: GT image, metadata (e.g. blur kernel params, motion, etc.)
# Output: Blurred image
# ------------------------------------------------------------------------
from diffusers import AutoencoderKL
import einops
import numpy as np
from ptlflow.utils import flow_utils
import torch

from models.archs import define_network
from models.base_model import BaseModel
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics
from models.archs.Encoder_arch import Encoder_arch, Encoder_config
from utils.misc import log_video

class Blur_model(BaseModel):
    """Model that takes GT (sharp) image and metadata as input and generates the blurred image."""

    def __init__(self, opt, logger):
        super(Blur_model, self).__init__(opt, logger)

        self.meta_key = getattr(opt, "meta_key", "meta")

        self.net_g = define_network(opt.network)
        self.models.append(self.net_g)
        self.vae = None
        if opt.vae_name_or_path:
            self.vae = AutoencoderKL.from_pretrained(opt.vae_name_or_path, subfolder=opt.vae_subfolder)
            self.vae.requires_grad_(False)
            self.vae.eval()
            self.vae.to(self.accelerator.device)
            self.models.append(self.vae)

        self.meta_vae = None
        if opt.meta_vae:
            if opt.meta_vae.type == 'AutoencoderKL':
                opt.meta_vae.pop('type')
                self.meta_vae = AutoencoderKL(**opt.meta_vae.__dict__)
            else:
                self.meta_vae = define_network(opt.meta_vae)
            self.models.append(self.meta_vae)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        self.optimizer_g = torch.optim.Adam([{'params': self.net_g.parameters()}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(self.optimizer_g)
        if self.meta_vae is not None:
            self.optimizer_meta_vae = torch.optim.Adam([{'params': self.meta_vae.parameters()}],
                lr=train_opt.optim.learning_rate,
                betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
                weight_decay=train_opt.optim.adam_weight_decay,
                eps=train_opt.optim.adam_epsilon,
            )
            self.optimizers.append(self.optimizer_meta_vae)
        else:
            self.optimizer_meta_vae = None

    def _get_meta_spatial(self, meta, h, w, gt_batch_size):
        """Convert metadata to spatial form (B, D, H, W) for concatenation with image."""
        if meta.dim() == 2:
            # (B, D) -> align batch with gt when patched (gt may be B*num_patches)
            if meta.shape[0] != gt_batch_size:
                meta = meta.repeat_interleave(gt_batch_size // meta.shape[0], dim=0)
            # (B, D) -> (B, D, 1, 1) -> (B, D, H, W)
            meta = meta.unsqueeze(-1).unsqueeze(-1)
            meta = meta.expand(-1, -1, h, w)
        elif meta.dim() != 4:
            raise ValueError(
                f"Metadata must be (B, D) or (B, D, H, W), got shape {meta.shape}"
            )
        return meta

    def _concat_gt_meta(self, gt, meta):
        """Concatenate GT image and metadata into network input."""
        b, c, h, w = gt.shape
        meta_spatial = self._get_meta_spatial(meta, h, w, b)
        return torch.cat([gt, meta_spatial], dim=1)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = (
            self.dataloader.dataset.gt_key
            if is_train
            else self.test_dataloader.dataset.gt_key
        )
        lq_key = (
            self.dataloader.dataset.lq_key
            if is_train
            else self.test_dataloader.dataset.lq_key
        )
        meta_key = (
            self.dataloader.dataset.meta_key
            if is_train
            else self.test_dataloader.dataset.meta_key
        )
        if self.opt.train.patched if is_train else self.opt.val.patched:
            # Only patch image-shaped tensors; metadata (B, D) is not patched
            keys = [gt_key, lq_key, meta_key]
            self.grids(keys=keys, opt=self.opt.train if is_train else self.opt.val)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        gt = self.sample[gt_key]
        lq_target = self.sample[lq_key]
        meta = self.sample[self.meta_key]

        all_losses = {}
        if self.meta_vae is not None:
            meta_enc = self.meta_vae.encode(meta).latent_dist.sample() * self.meta_vae.config.scaling_factor
            meta_dec = self.meta_vae.decode(meta_enc/self.meta_vae.config.scaling_factor).sample
            losses = self.criterion(meta_dec, meta.detach())
            self.accelerator.backward(losses["all"])
            torch.nn.utils.clip_grad_norm_(self.meta_vae.parameters(), 1.0)
            self.optimizer_meta_vae.step()
            all_losses["meta"] = losses["all"]
        else:
            meta_enc = meta
        if self.vae is not None:
            gt = self.vae.encode(gt).latent_dist.sample() * self.vae.config.scaling_factor
            lq_target = self.vae.encode(lq_target).latent_dist.sample() * self.vae.config.scaling_factor
        
        net_in = self._concat_gt_meta(gt, meta_enc.detach())
        pred_blur = self.net_g(net_in)

        losses = self.criterion(pred_blur, lq_target.detach())
        all_losses["gen"] = losses["all"]
        self.accelerator.backward(losses["all"])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1.0)
        self.optimizer_g.step()

        return all_losses

    @torch.no_grad()
    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        gt = self.sample[gt_key]
        lq_target = self.sample[lq_key]

        if self.opt.val.patched:
            b, c, h, w = self.original_size[gt_key]
            pred_list = []
            meta_list = []
            for _ in self.setup_patches():
                _gt = self.sample[gt_key]
                _meta = self.sample[self.meta_key]
                if self.meta_vae is not None:
                    _meta_enc = self.meta_vae.encode(_meta).latent_dist.sample() * self.meta_vae.config.scaling_factor
                    _meta_dec = self.meta_vae.decode(_meta_enc/self.meta_vae.config.scaling_factor).sample
                    meta_list.append(_meta_dec)
                else:
                    _meta_enc = _meta
                if self.vae is not None:
                    _gt = self.vae.encode(_gt).latent_dist.sample() * self.vae.config.scaling_factor
                net_in = self._concat_gt_meta(_gt, _meta_enc)
                pred = self.net_g(net_in)
                if self.vae is not None:
                    pred = self.vae.decode(pred/self.vae.config.scaling_factor).sample
                pred_list.append(pred)
            pred = torch.cat(pred_list, dim=0)
            pred = einops.rearrange(pred, "(b n) c h w -> b n c h w", b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[gt_key + "_patched_pos"])
                out.append(merged[..., :h, :w])
            pred = torch.stack(out)

            if self.meta_vae is not None:
                meta_dec = torch.cat(meta_list, dim=0)
                meta_dec = einops.rearrange(meta_dec, "(b n) c h w -> b n c h w", b=b)
                out = []
                for i in range(len(meta_dec)):
                    merged = merge_patches(meta_dec[i], self.sample[gt_key + "_patched_pos"])
                    out.append(merged[..., :h, :w])
                meta_dec = torch.stack(out)
    
            gt = self.sample[gt_key + "_original"]
            lq_target = self.sample[lq_key + "_original"]
            meta = self.sample[self.meta_key + "_original"]
        else:
            meta = self.sample[self.meta_key]
            if self.meta_vae is not None:
                _meta_enc = self.meta_vae.encode(meta).latent_dist.sample() * self.meta_vae.config.scaling_factor
                meta_dec = self.meta_vae.decode(_meta/self.meta_vae.config.scaling_factor).sample
            else:
                _meta_enc = meta
            if self.vae is not None:
                _gt = self.vae.encode(gt).latent_dist.sample() * self.vae.config.scaling_factor
            else:
                _gt = gt
            net_in = self._concat_gt_meta(_gt, _meta_enc)
            pred = self.net_g(net_in)
            if self.vae is not None:
                pred = self.vae.decode(pred/self.vae.config.scaling_factor).sample

        k_bg = np.clip(self.sample[self.meta_key+"_kernel_bg"].cpu().numpy(), 0, 1)
        k_fg = np.clip(self.sample[self.meta_key+"_kernel_fg"].cpu().numpy(), 0, 1)
        meta = einops.rearrange(meta, 'b (f n) h w -> b f n h w', n=2)
        if self.meta_vae is not None:
            meta_dec = einops.rearrange(meta_dec, 'b (f n) h w -> b f n h w', n=2)
        
        gt = gt.cpu().numpy() * 0.5 + 0.5
        lq_target = lq_target.cpu().numpy() * 0.5 + 0.5
        pred = pred.cpu().numpy() * 0.5 + 0.5

        for i in range(len(gt)):
            idx += 1
            # Log: [GT sharp | pred blur | target blur]
            image1 = einops.rearrange(
                [gt[i], pred[i], lq_target[i]], "n c h w -> c h (n w)"
            )
            image1 = np.clip(image1[np.newaxis, ...], 0, 1)
            log_image(
                self.opt,
                self.accelerator,
                image1,
                f"{idx:04d}",
                self.global_step,
            )
            log_image(
                self.opt,
                self.accelerator,
                np.clip(pred[i : i + 1], 0, 1),
                f"pred_blur_{idx:04d}",
                self.global_step,
            )
            log_image(
                self.opt,
                self.accelerator,
                k_bg[i : i + 1, None]/k_bg[i].max(),
                f"k_bg_{idx:04d}",
                self.global_step,
            )
            log_image(
                self.opt,
                self.accelerator,
                k_fg[i : i + 1, None]/k_fg[i].max(),
                f"k_fg_{idx:04d}",
                self.global_step,
            )
            log_metrics(
                lq_target[i],
                pred[i],
                self.opt.val.metrics,
                self.accelerator,
                self.global_step,
            )

            gt_flow_video = []
            for j in range(meta.shape[1]):
                gt_flow_video.append(flow_utils.flow_to_rgb(meta[i, j].to(torch.float32)))
            gt_flow_video =  torch.stack(gt_flow_video, dim=0).cpu().numpy().transpose([0, 2, 3, 1])
            log_video(
                self.opt,
                self.accelerator,
                gt_flow_video,
                f"flow_{idx:04d}",
                self.global_step,
            )
            if self.meta_vae is not None:
                meta_dec_flow_video = []
                for j in range(meta_dec.shape[1]):
                    meta_dec_flow_video.append(flow_utils.flow_to_rgb(meta_dec[i, j].to(torch.float32)))
                meta_dec_flow_video = torch.stack(meta_dec_flow_video, dim=0).cpu().numpy().transpose([0, 2, 3, 1])
                log_video(
                    self.opt,
                    self.accelerator,
                    meta_dec_flow_video,
                    f"dec_flow_{idx:04d}",
                    self.global_step,
                )
        return idx

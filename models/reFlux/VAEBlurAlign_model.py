"""VAEBlurAlign_model: LoRA-adapt the FLUX.2 Klein VAE's ENCODER (decoder stays
frozen) with a JEPA-style block-replacement objective, per papers/main.tex
Sec. 5/6 and the follow-up discussion that corrected this file's first draft.

This mimics how JEPA actually trains, not "encode two separate full images and
match them everywhere": FullResBlurDataset gives one MIXED image per sample --
GT everywhere except one rectangular block, which is replaced by its blurred
counterpart (blur substitutes for JEPA's mask token, since the corrupted
region here is degraded rather than removed). Two SEPARATE encoders process
the two images -- this is not the same module called twice:

  1. encode(mixed) via self.vae     -> z_mixed  (trainable: LoRA gradients flow)
  2. encode(gt)    via self.vae_gt  -> z_gt     (a second, fully frozen copy of
                                                  the pretrained VAE encoder --
                                                  no LoRA, requires_grad=False,
                                                  called under torch.no_grad())

Using the SAME LoRA-adapted encoder for both (with only a stop-gradient on the
target branch) was tried first and didn't work: the target's *forward value*
still drifted every step as the shared LoRA weights updated, so the encoder
was chasing a moving target it was itself moving -- this line of the code
produced a fast, noisy, non-decreasing loss plateau (see experiment_plan.html
for the checkpoint-diff / validation-image evidence that caught it). A truly
frozen target network removes that feedback loop entirely: z_gt is a fixed
function of the pretrained weights for the whole run.

The latent loss is computed ONLY at the token positions inside the replaced
block (the dataset's per-pixel mask, downsampled to the packed-token grid) --
exactly analogous to JEPA computing its loss only on masked/target patches,
using the surrounding clean context implicitly available to the (spatially
non-local, convolutional) encoder. A second, pixel-space MSE loss on the
decoded image (also cropped to the replaced block) backs this up: it's the
same latent-matching objective but after the frozen decoder, so it can catch
latent-space errors the token-level MSE might undervalue.

A third, adversarial term addresses a failure mode the two MSE losses can't
see: validation showed the decoded output had checkerboard artifacts and was
measurably *worse* than just leaving the input alone (block-region MSE vs GT
higher for the prediction than for the untouched blurry input, in 11/11
sampled cases) -- likely because removing the old full-frame reconstruction
anchor left nothing constraining the encoder's general decode fidelity, only
the two block-local MSE terms. A PatchGAN discriminator (reused from
models/archs/related/LD/MultiscaleDiscriminator_arch.py, the same one the
KleinDistill family trains) judges the FULL decoded frame against real GT
images -- unconditional, not cropped to the block, since checkerboarding is a
decoder-fidelity problem visible everywhere, not just in the replaced region.
Discriminator params are frozen (requires_grad=False) during the generator's
forward pass through D so that adversarial-loss backward doesn't also dirty
D's gradients before its own optimizer step; both G and D updates are gated
on self.accelerator.sync_gradients so D accumulates over the same effective
batch as G rather than updating 4x more often under gradient accumulation.

Only the encoder's LoRA parameters (and the discriminator) train. The VAE
decoder is never touched, so `decode_packed_vae_tokens` always uses the exact
same frozen mapping Stage B and everything else in this repo already assumes.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2

from dataset import create_dataset
from models.archs.related.LD.MultiscaleDiscriminator_arch import MultiscaleDiscriminator
from models.archs.related.LD.loss import GANLoss
from models.base_model import BaseModel
from models.LucidFlux.helpers import (
    decode_packed_vae_tokens,
    encode_vae_packed_tokens,
    get_flux2_pipeline_class,
)
from utils import log_image, log_metrics


def _cfg(opt, key, default=None):
    value = opt.get(key, default) if hasattr(opt, "get") else getattr(opt, key, default)
    return default if value in (None, "~") else value


def lora_adapt_vae_encoder(vae, rank: int):
    """Add a LoRA adapter to the VAE's ENCODER modules only (conv + attention
    projections under any name containing "encoder", plus quant_conv but not
    post_quant_conv), leaving every decoder-side weight untouched.

    Same target-module selection as Flux2DeblurCycle_model.initialize_vae();
    duplicated (not imported) so this file has no cross-family dependency on
    models/DeblurDeform.
    """
    target_modules = []
    grep_patterns = [
        "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
        "to_k", "to_q", "to_v", "to_out.0",
    ]
    for n, _p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in grep_patterns:
            if pattern in n and "encoder" in n:
                target_modules.append(n.replace(".weight", ""))
                break
        else:
            if "quant_conv" in n and "post_quant_conv" not in n:
                target_modules.append(n.replace(".weight", ""))

    lora_conf = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=target_modules)
    vae.add_adapter(lora_conf, adapter_name="default_encoder")
    vae.set_adapter(["default_encoder"])
    return vae


class VAEBlurAlign_model(BaseModel):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.lora_rank = int(_cfg(opt, "lora_rank", 4))
        self.latent_loss_weight = float(_cfg(opt, "latent_loss_weight", 1.0))
        self.pixel_loss_weight = float(_cfg(opt, "pixel_loss_weight", 1.0))
        # Default kept low: this repo's own KleinDistill family hit a
        # discriminator-overpowers-generator collapse at lambda_gan=0.1 with
        # matched G/D learning rates (see Flux2KleinHybrid_model.py) and
        # settled on 0.02.
        self.gan_loss_weight = float(_cfg(opt, "gan_loss_weight", 0.02))
        self.gan_mode = _cfg(opt, "gan_mode", "hinge")

        pretrained = opt.pretrained_model_name_or_path
        revision = _cfg(opt, "revision", None)
        variant = _cfg(opt, "variant", None)
        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)

        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.vae.requires_grad_(False)
        self.vae = lora_adapt_vae_encoder(self.vae, self.lora_rank)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)
        # The pixel/GAN losses backprop through the frozen decoder, which is
        # memory-heavy -- same mitigation ContextProjectorAlign_model uses.
        if self.is_train and (self.pixel_loss_weight > 0 or self.gan_loss_weight > 0):
            self.vae.enable_gradient_checkpointing()
            self.vae.enable_slicing()
        self.models.append(self.vae)

        # Separate, permanently frozen encoder for the GT target -- a second
        # full copy of the pretrained weights, no LoRA, never in self.models
        # (so the train()/eval() toggling around validation() never touches
        # it, and it's never handed to setup_optimizers()). This is what makes
        # z_gt a fixed target instead of one that drifts with self.vae's LoRA
        # updates.
        self.vae_gt = AutoencoderKLFlux2.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.vae_gt.requires_grad_(False)
        self.vae_gt = self.vae_gt.to(self.accelerator.device, dtype=weight_dtype)
        self.vae_gt.eval()

        self.discriminator = None
        self.criterionGAN = None
        if self.is_train and self.gan_loss_weight > 0:
            # input_nc=3, unconditional -- judges the decoded RGB frame on its
            # own against real GT frames, not conditioned on any input (the
            # MultiscaleDiscriminator_arch auto-discovered wrapper hardcodes
            # input_nc=6 for LQ+HQ-conditional setups, so this imports the
            # underlying MultiscaleDiscriminator class directly instead).
            # Kept in float32 (not weight_dtype/bf16): spectral norm's power
            # iteration is a small extra cost either way, and this is a tiny
            # network, so there's no reason to trade its numerical stability
            # for bf16 memory/speed here.
            self.discriminator = MultiscaleDiscriminator(input_nc=3).to(self.accelerator.device)
            self.models.append(self.discriminator)
            self.criterionGAN = GANLoss(gan_mode=self.gan_mode)

        n_all = sum(p.numel() for p in self.vae.parameters())
        n_trainable = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        logger.info(
            "VAEBlurAlign: lora_rank=%d trainable=%d/%d (%.2f%%) "
            "frozen_gt_encoder=separate_copy loss_weights(latent=%.2f pixel=%.2f gan=%.3f[%s])",
            self.lora_rank, n_trainable, n_all,
            100 * n_trainable / n_all,
            self.latent_loss_weight, self.pixel_loss_weight, self.gan_loss_weight, self.gan_mode,
        )

    def setup_optimizers(self):
        opt = self.opt.train.optim
        trainable_params = [p for p in self.vae.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=_cfg(opt, "adam_weight_decay", 0.01),
            eps=_cfg(opt, "adam_epsilon", 1e-8),
        )
        self.optimizers.append(optimizer)  # index 0: generator (VAE LoRA)

        if self.discriminator is not None:
            d_lr = float(_cfg(opt, "discriminator_learning_rate", opt.learning_rate))
            optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=d_lr,
                betas=(opt.adam_beta1, opt.adam_beta2),
                weight_decay=_cfg(opt, "adam_weight_decay", 0.01),
                eps=_cfg(opt, "adam_epsilon", 1e-8),
            )
            self.optimizers.append(optimizer_d)  # index 1: discriminator

    def setup_dataloaders(self):
        # FullResPatchifierDataset (and any other IterableDataset here) yields
        # already-batched tensors itself -- each __iter__ step produces a
        # full [batch_size, C, H, W] dict, since samples can only be
        # torch.stack'd if they share one crop shape, which the dataset
        # enforces by fixing one shared shape per yielded group rather than
        # per dataloader-level batch_size. DataLoader(batch_size=...) would
        # instead try to stack N of those already-batched dicts on top of
        # each other, adding a spurious leading dim. batch_size=None disables
        # that: the loader just forwards what __iter__ produces.
        from torch.utils.data import DataLoader, IterableDataset

        self.logger.info("Setting up val dataloaders...")
        val_set = create_dataset(self.opt.datasets.val)
        val_iterable = isinstance(val_set, IterableDataset)
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=False if val_iterable else self.opt.datasets.val.use_shuffle,
            batch_size=None if val_iterable else self.opt.val.batch_size,
            num_workers=self.opt.datasets.val.get('num_worker_per_gpu', 0),
        )

        self.logger.info("Setting up train dataloader...")
        train_set = create_dataset(self.opt.datasets.train)
        train_iterable = isinstance(train_set, IterableDataset)
        self.dataloader = DataLoader(
            train_set,
            shuffle=False if train_iterable else self.opt.datasets.train.use_shuffle,
            batch_size=None if train_iterable else self.opt.train.batch_size,
            num_workers=self.opt.datasets.train.get('num_worker_per_gpu', 0),
        )

    def feed_data(self, data, is_train=True):
        self.sample = data

    def _encode(self, image_pm1: torch.Tensor):
        """Returns (packed_tokens [B,N,C], patchified_latents) -- gradients flow
        normally (no no_grad/inference_mode wrapper); callers detach as needed."""
        return encode_vae_packed_tokens(
            self.vae, image_pm1.to(self.weight_dtype), flux2_pipe_cls=self.flux2_pipe_cls
        )

    def _decode(self, packed_tokens, patchified_latents):
        return decode_packed_vae_tokens(
            packed_tokens, patchified_latents, self.vae, flux2_pipe_cls=self.flux2_pipe_cls
        )

    @torch.no_grad()
    def _encode_gt(self, image_pm1: torch.Tensor):
        """Frozen target encoder -- separate weights from self.vae, no LoRA,
        no gradient ever. Returns (packed_tokens, patchified_latents), both
        already detached by construction (no_grad)."""
        return encode_vae_packed_tokens(
            self.vae_gt, image_pm1.to(self.weight_dtype), flux2_pipe_cls=self.flux2_pipe_cls
        )

    @staticmethod
    def _token_mask(pixel_mask: torch.Tensor, patchified_latents: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] pixel-space replacement mask -> [B,N_slots,1] boolean mask
        over packed tokens, matching _pack_latents' row-major (h,w) flatten."""
        grid_h, grid_w = patchified_latents.shape[-2:]
        pooled = F.adaptive_avg_pool2d(pixel_mask.float(), (grid_h, grid_w))
        token_mask = (pooled > 0.5).float()  # [B,1,grid_h,grid_w]
        token_mask = token_mask.reshape(token_mask.shape[0], 1, grid_h * grid_w)
        return token_mask.permute(0, 2, 1)  # [B, N_slots, 1]

    @staticmethod
    def _mask_bbox(mask_2d: torch.Tensor):
        """[H,W] binary mask -> (top, bottom, left, right) of its bounding box."""
        ys, xs = torch.nonzero(mask_2d > 0.5, as_tuple=True)
        if ys.numel() == 0:
            return 0, mask_2d.shape[0], 0, mask_2d.shape[1]
        return (
            int(ys.min()), int(ys.max()) + 1,
            int(xs.min()), int(xs.max()) + 1,
        )

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        mask_key = self.dataloader.dataset.mask_key
        gt_image = self.sample[gt_key].clip(-1, 1)
        mixed_image = self.sample[lq_key].clip(-1, 1)
        pixel_mask = self.sample[mask_key]

        # Pass 1: mixed context, through the trainable (LoRA) encoder.
        z_mixed, mixed_patchified = self._encode(mixed_image)
        # Pass 2: GT, through the separate FROZEN target encoder (self.vae_gt)
        # -- a fixed function of the pretrained weights, never the shared
        # LoRA-adapted one, so it can't drift as self.vae trains.
        z_gt, gt_patchified = self._encode_gt(gt_image)

        token_mask = self._token_mask(pixel_mask, mixed_patchified)  # [B,N,1]
        n_masked = token_mask.sum().clamp(min=1.0)

        # z_gt is already gradient-free (produced under no_grad by a frozen
        # encoder) -- no separate stop-grad step needed here.
        # Masked MSE: only token positions inside the replaced block are
        # supervised -- positions outside it already have input == GT, so
        # they get no direct latent loss here (matches JEPA: no loss on
        # visible/context patches). Normalize by (masked tokens * channels).
        sq_err = (z_mixed.float() - z_gt.float()) ** 2
        loss_latent = (sq_err * token_mask).sum() / (n_masked * z_mixed.shape[-1])

        losses = {"loss_latent": loss_latent, "all": self.latent_loss_weight * loss_latent}

        pred_image = None
        if self.pixel_loss_weight > 0 or self.gan_loss_weight > 0:
            pred_image = self._decode(z_mixed, mixed_patchified)

        if self.pixel_loss_weight > 0:
            # Pixel-space MSE, cropped to the replaced block. Computed over
            # the whole frame, ~50-70% of every pixel is already-correct GT
            # context (input == target there already), so most of the
            # gradient would just reward "keep doing what you're already
            # doing" instead of pushing on the one region that actually needs
            # to change -- this was diluting the signal enough that training
            # visibly plateaued with no change in decoded output (see
            # experiment_plan.html for the checkpoint comparison that caught
            # this).
            # Each sample in the batch has its OWN independently-sampled
            # block (FullResPatchifierDataset draws the block per-image, not
            # per-batch) -- can't crop the whole batched tensor with one
            # bbox and call mse_loss once, since different samples' boxes
            # have different sizes and can't share a single slice. Loop
            # per-sample and average, same pattern validate_step already uses
            # for its per-sample metrics below.
            bsz = pred_image.shape[0]
            loss_pixel = 0.0
            for i in range(bsz):
                t, b, l, r = self._mask_bbox(pixel_mask[i, 0])
                loss_pixel = loss_pixel + F.mse_loss(
                    pred_image[i:i + 1, :, t:b, l:r].float(), gt_image[i:i + 1, :, t:b, l:r].float()
                )
            loss_pixel = loss_pixel / bsz
            losses["loss_pixel"] = loss_pixel
            losses["all"] = losses["all"] + self.pixel_loss_weight * loss_pixel

        if self.gan_loss_weight > 0:
            # Generator step: freeze D's params (requires_grad=False) for this
            # forward pass so the adversarial term's backward doesn't also
            # deposit gradient onto D's weights -- D's own gradient has to
            # come ONLY from its own real/fake forward+backward below, or
            # this contaminates it before optimizer[1].step() runs.
            for p in self.discriminator.parameters():
                p.requires_grad = False
            fake_pred_g = self.discriminator(pred_image.float())
            loss_gan = self.criterionGAN(fake_pred_g, True, for_discriminator=False)
            losses["loss_gan"] = loss_gan
            losses["all"] = losses["all"] + self.gan_loss_weight * loss_gan
            for p in self.discriminator.parameters():
                p.requires_grad = True

        self.accelerator.backward(losses["all"])

        if self.gan_loss_weight > 0:
            # Discriminator step: real=GT, fake=decoded pred (detached -- no
            # generator gradient here, this backward is D-only).
            real_pred = self.discriminator(gt_image.float().detach())
            fake_pred_d = self.discriminator(pred_image.float().detach())
            loss_d = 0.5 * (
                self.criterionGAN(real_pred, True, for_discriminator=True)
                + self.criterionGAN(fake_pred_d, False, for_discriminator=True)
            )
            self.accelerator.backward(loss_d)
            losses["loss_d"] = loss_d

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                [p for p in self.vae.parameters() if p.requires_grad],
                getattr(self.opt.train.optim, "max_grad_norm", 1.0),
            )
            self.optimizers[0].step()
            self.optimizers[0].zero_grad()
            # D and G updates both gate on sync_gradients so D accumulates
            # over the same effective batch as G, rather than updating on
            # every micro-batch while G only updates once per
            # gradient_accumulation_steps micro-batches.
            if self.gan_loss_weight > 0:
                self.accelerator.clip_grad_norm_(
                    self.discriminator.parameters(),
                    getattr(self.opt.train.optim, "max_grad_norm", 1.0),
                )
                self.optimizers[1].step()
                self.optimizers[1].zero_grad()

        return losses

    @torch.no_grad()
    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        mask_key = self.test_dataloader.dataset.mask_key
        gt_image = self.sample[gt_key].clip(-1, 1)
        mixed_image = self.sample[lq_key].clip(-1, 1)
        pixel_mask = self.sample[mask_key]
        bsz = gt_image.shape[0]

        # Both panels go through the TRAINABLE encoder (self.vae) -- this is
        # the only encoder that exists at inference time; vae_gt is
        # training-time-only scaffolding for the frozen target and is never
        # used here, so "recon" reflects the same recon-anchor pathway that's
        # actually being trained above.
        z_mixed, mixed_patchified = self._encode(mixed_image)
        z_gt, gt_patchified = self._encode(gt_image)

        pred_image = self._decode(z_mixed, mixed_patchified).clamp(-1, 1)
        recon_image = self._decode(z_gt, gt_patchified).clamp(-1, 1)

        val_metrics = getattr(self.opt.val, "metrics", None)
        if val_metrics and self.accelerator.is_main_process:
            # log_metrics (via calculate_ssim's cv2 calls) expects a single
            # [H,W,C] image, not a batch -- loop per-sample, matching every
            # other model in this repo (e.g. Simple_model.validate_step).
            gt_np = (gt_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            mixed_np = (mixed_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            pred_np = (pred_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            recon_np = (recon_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            mask_np = pixel_mask.cpu().numpy()
            for i in range(bsz):
                # Full-frame numbers: mostly diagnostic now, since most pixels
                # are literally GT (unmodified) -- "floor_" no longer means
                # "the whole image is blurred," just "before any recovery."
                log_metrics(gt_np[i], pred_np[i], val_metrics, self.accelerator, self.global_step)
                log_metrics(gt_np[i], mixed_np[i], val_metrics, self.accelerator, self.global_step, comment="floor_")
                log_metrics(gt_np[i], recon_np[i], val_metrics, self.accelerator, self.global_step, comment="reconanchor_")

                # The number that actually answers the question: PSNR/SSIM
                # restricted to the replaced block only (input vs. recovered).
                # gt_np[i] is [C,H,W] -- slice H,W (axes 1,2), not C first.
                # crop=None here (NOT 0: calculate_mse/psnr slice img[crop:-crop],
                # and 0:-0 is an empty array, not "no crop") -- these bbox crops
                # can be much smaller than a full 512-frame, so the frame-level
                # border crop (30px) used for the other metrics above would be
                # wrong (or empty) here.
                ys, xs = np.nonzero(mask_np[i, 0] > 0.5)
                if len(ys) > 0:
                    t, b, l, r = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    block_metrics = {k: {"crop": None} for k in val_metrics}
                    log_metrics(
                        gt_np[i][:, t:b, l:r], mixed_np[i][:, t:b, l:r],
                        block_metrics, self.accelerator, self.global_step, comment="blockfloor_",
                    )
                    log_metrics(
                        gt_np[i][:, t:b, l:r], pred_np[i][:, t:b, l:r],
                        block_metrics, self.accelerator, self.global_step, comment="blockpred_",
                    )

        if getattr(self.opt.val, "save_images", False) and self.accelerator.is_main_process:
            # Frozen decoder only, for every decoded panel -- the LoRA'd
            # encoder never touches decode(), so this is exactly what a
            # downstream consumer of this checkpoint would see.
            gt_01 = (gt_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            mixed_01 = (mixed_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            pred_01 = (pred_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            recon_01 = (recon_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            mask_01 = np.repeat(pixel_mask.cpu().numpy().clip(0, 1), 3, axis=1)
            for i in range(bsz):
                idx += 1
                panels = np.stack([mixed_01[i], pred_01[i], recon_01[i], gt_01[i], mask_01[i]])
                log_image(
                    self.opt, self.accelerator, panels,
                    f"{idx:04d}_mixed_pred_reconanchor_gt_mask", self.global_step,
                )
        else:
            idx += bsz

        return idx

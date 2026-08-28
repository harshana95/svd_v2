"""Rectified-Flow (ReFlow) distillation of FLUX.2-klein-4B for pure text-to-image
generation (no LQ/reference-image conditioning).

Same reflow recipe as `Flux2KleinReflow_model` (Liu et al., "Flow Straight and
Fast"): the frozen teacher is rolled out over `num_teacher_steps` Euler steps
from noise to build a self-consistent (noise, output) pair, and the student is
trained with a flow-matching MSE loss on the straight line between that pair,
plus a pixel-anchor loss (L2 + LPIPS) from the student's own
`num_student_steps` rollout against a real image. Unlike the restoration
variant, there is no reference image to condition on — the transformer only
ever sees the noisy latent tokens, and the text branch is fed the dataset's
real captions (encoded fresh every step) instead of a fixed null prompt. This
is closer to the reflow paper's original text-to-image setting than the
restoration adaptation.

GAN branch: MultiscaleDiscriminator normally expects a 6-channel (condition,
image) pair, which doesn't exist here — instantiated with input_nc=3 instead
(unconditional, classifies the generated/real image alone) rather than the
restoration models' lq||image concatenation. Without it, 1-step distillation
here collapsed to a near-uniform blurry output over training (pure L2+LPIPS
supervision on a single Euler step has no pressure against regressing to a
low-frequency average); the adversarial term is what's supposed to fix that.
"""

import gc
import random
import re

import einops
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.base_model import BaseModel
from models.archs.related.LD.MultiscaleDiscriminator_arch import MultiscaleDiscriminator
from models.archs.related.LD.loss import GANLoss
from utils import log_image, log_metrics

from .helpers import (
    calculate_shift,
    decode_latents_flux,
    get_flux2_pipeline_class,
    initialize_transformer_lora,
)

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2


class Flux2KleinReflowT2I_model(BaseModel):
    """ReFlow distillation of FLUX.2-klein-4B into a `num_student_steps`-step
    text-to-image student.

    Components:
      transformer     – student with LoRA (trainable, optimizers[0])
      transformer_fix – frozen teacher (fp16), used only under no_grad to build
                        reflow (noise, output) pairs
      vae             – frozen encoder/decoder
      net_lpips       – frozen perceptual loss
      text_pipe       – frozen text encoder(s) + tokenizer only (no transformer/
                        vae loaded), used to encode real captions every step
      net_d           – unconditional multiscale discriminator (optimizers[1])
    """

    def __init__(self, opt, logger):
        super().__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        pretrained = str(opt.pretrained_model_name_or_path)
        revision = opt.get("revision", None)
        variant = opt.get("variant", None)

        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)
        self.guidance_scale = float(opt.get("guidance_scale", 1.0))
        self.discrete_flow_shift = float(opt.get("discrete_flow_shift", 3.1582))

        # `opt`/`opt.train` are DictAsMember: __getattr__ returns None for missing
        # keys instead of raising AttributeError, which silently breaks Python's
        # getattr(obj, key, default) fallback. Use .get() everywhere in this file.
        self.lambda_reflow = float(opt.train.get("lambda_reflow", 1.0))
        self.lambda_l2 = float(opt.train.get("lambda_l2", 1.0))
        self.lambda_lpips = float(opt.train.get("lambda_lpips", 1.0))
        self.lambda_gan = float(opt.train.get("lambda_gan", 0.1))
        self.num_teacher_steps = int(opt.train.get("num_teacher_steps", 8))
        self.num_student_steps = int(opt.train.get("num_student_steps", 1))
        self.default_h = int(opt.train.get("image_h", 512))
        self.default_w = int(opt.train.get("image_w", 512))
        lora_rank = int(opt.train.get("lora_rank", 16))
        lora_alpha = int(opt.train.get("lora_alpha", lora_rank * 2))
        lora_dropout = float(opt.train.get("lora_dropout", 0.0))
        lora_target_modules = opt.train.get("lora_target_modules", None)

        # ── VAE ──────────────────────────────────────────────────────────────
        self.vae = AutoencoderKLFlux2.from_pretrained(pretrained, subfolder="vae", revision=revision, variant=variant)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ── Noise scheduler (config template for building per-rollout schedulers) ──
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")

        # ── Student transformer (LoRA) ────────────────────────────────────────
        base_transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant
        )
        base_transformer.requires_grad_(False)
        if opt.get("transformer_gradient_checkpointing", False):
            if hasattr(base_transformer, "enable_gradient_checkpointing"):
                base_transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled for student transformer")

        self.transformer = initialize_transformer_lora(
            base_transformer, rank=lora_rank, alpha=lora_alpha,
            dropout=lora_dropout, target_patterns=lora_target_modules,
        )
        for name, param in self.transformer.named_parameters():
            param.requires_grad = "lora" in name
        self.use_transformer_guidance = getattr(self.transformer.config, "guidance_embeds", True)

        # ── Frozen teacher (fix) — fp16 to save VRAM, only ever run under no_grad ──
        self.transformer_fix = Flux2Transformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant,
            torch_dtype=torch.float16,
        )
        self.transformer_fix.requires_grad_(False)
        self.transformer_fix.eval()

        # ── Perceptual loss ───────────────────────────────────────────────────
        self.net_lpips = lpips.LPIPS(net="vgg")
        self.net_lpips.requires_grad_(False)

        # ── Discriminator (unconditional multiscale PatchGAN — input_nc=3, no
        # LQ image to condition on, unlike the restoration models' input_nc=6) ──
        self.net_d = MultiscaleDiscriminator(input_nc=3)
        self.criterionGAN = GANLoss("hinge")

        # ── Text encoder(s) only — encodes real captions fresh every step, so it
        # stays resident (unlike the restoration model's one-off null-embedding cache) ──
        self.text_pipe = self.flux2_pipe_cls.from_pretrained(
            pretrained, transformer=None, vae=None, revision=revision, variant=variant,
            torch_dtype=weight_dtype,
        )

        # ── Move to device ────────────────────────────────────────────────────
        device = self.accelerator.device
        self.transformer = self.transformer.to(device, dtype=weight_dtype)
        self.transformer_fix = self.transformer_fix.to(device, dtype=torch.float16)
        self.vae = self.vae.to(device, dtype=weight_dtype)
        self.net_lpips = self.net_lpips.to(device)
        self.text_pipe = self.text_pipe.to(device)
        self.net_d = self.net_d.to(device)

        # student and discriminator are tracked for checkpointing.
        self.models.append(self.transformer)
        self.models.append(self.net_d)

        total = sum(p.numel() for p in self.transformer.parameters())
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        logger.info(f"{self.transformer.__class__.__name__}: trainable={trainable:,} / total={total:,} ({100*trainable/total:.3f}%)")
        logger.info(f"ReFlow T2I distillation: teacher steps={self.num_teacher_steps}, student steps={self.num_student_steps}")

    # ── Text embeddings ────────────────────────────────────────────────────────

    def _encode_captions(self, captions, device):
        with torch.no_grad():
            prompt_embeds, text_ids = self.text_pipe.encode_prompt(list(captions), num_images_per_prompt=1)
        txt = prompt_embeds.to(device=device, dtype=self.weight_dtype)
        text_ids = text_ids.to(device=device, dtype=self.weight_dtype)
        if text_ids.ndim == 2:
            text_ids = text_ids.unsqueeze(0).expand(txt.shape[0], -1, -1)
        return txt, text_ids

    def _transformer_guidance(self, batch_size: int, device):
        if not self.use_transformer_guidance:
            return None
        return torch.full((batch_size,), self.guidance_scale, device=device, dtype=self.weight_dtype)

    # ── Discriminator helpers ─────────────────────────────────────────────────

    def _discriminate(self, fake_image: torch.Tensor, real_image: torch.Tensor):
        """Run the unconditional discriminator on fake and real images jointly.

        Both pass through in the same batch so BatchNorm sees consistent
        statistics (same pattern as the restoration models' _discriminate,
        minus the lq_image conditioning channel).
        """
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        disc_out = self.net_d(fake_and_real)
        pred_fake, pred_real = [], []
        for scale_out in disc_out:
            pred_fake.append([t[: t.size(0) // 2] for t in scale_out])
            pred_real.append([t[t.size(0) // 2 :] for t in scale_out])
        return pred_fake, pred_real

    def _update_discriminator(self, fake_image: torch.Tensor, real_image: torch.Tensor):
        """One discriminator update step. fake_image must already be detached."""
        if self.lambda_gan == 0.0:
            return fake_image.new_tensor(0.0)
        pred_fake_d, pred_real_d = self._discriminate(fake_image, real_image)
        loss_d_fake = self.criterionGAN(pred_fake_d, False, for_discriminator=True).mean()
        loss_d_real = self.criterionGAN(pred_real_d, True, for_discriminator=True).mean()
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.accelerator.backward(loss_d)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.disc_params, 1.0)
        self.optimizers[1].step()
        self.optimizers[1].zero_grad()
        return loss_d

    # ── Optimizers ────────────────────────────────────────────────────────────

    def setup_optimizers(self):
        opt = self.opt.train.optim
        use_8bit = opt.get("use_8bit_adam", False)
        if use_8bit:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.AdamW

        common_kwargs = dict(
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.get("adam_weight_decay", 0.01),
            eps=opt.get("adam_epsilon", 1e-8),
        )

        self.gen_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizers.append(optimizer_class(self.gen_params, **common_kwargs))

        self.disc_params = list(self.net_d.parameters())
        d_lr = opt.get("d_learning_rate", opt.learning_rate)
        self.optimizers.append(optimizer_class(self.disc_params, lr=d_lr,
                                               betas=(opt.adam_beta1, opt.adam_beta2),
                                               weight_decay=opt.get("adam_weight_decay", 0.01),
                                               eps=opt.get("adam_epsilon", 1e-8)))

    # ── Data feeding ──────────────────────────────────────────────────────────

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[gt_key], opt=self.opt.val)

    def _pick_captions(self):
        # 'caption' is a list of 5 human captions per image (flickr30k); the
        # default collate transposes that into 5 tuples of `batch_size`
        # strings. Pick one caption index for the whole batch each step for
        # some prompt variety across steps without needing a dataset-side
        # transform (captions are cheap to re-pick; images aren't).
        captions_by_index = self.sample["caption"]
        idx = random.randrange(len(captions_by_index))
        return list(captions_by_index[idx])

    # ── Shared K-step Euler rollout ───────────────────────────────────────────

    def _rollout(self, transformer, txt, txt_ids, lat_h, lat_w, bsz, device, dtype,
                 num_steps, grad_last_step_only=False):
        """Run `num_steps` Euler steps from pure noise through `transformer`,
        conditioned only on `txt`/`txt_ids` (no reference-image tokens).

        Returns (x0_final, z1_initial_noise). By default every step runs under
        no_grad (used for the frozen teacher, and for plain inference). When
        `grad_last_step_only` is True, only the final step keeps gradients, so a
        downstream loss can backprop into the student's parameters without
        holding activations for every discretization step in the rollout.
        """
        lat_c = transformer.config.in_channels
        z1 = torch.randn(bsz, lat_c, lat_h, lat_w, device=device, dtype=dtype)
        latents = z1

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.noise_scheduler.config)
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            image_seq_len = (lat_h // 2) * (lat_w // 2)
            mu = calculate_shift(
                image_seq_len,
                base_seq_len=getattr(scheduler.config, "base_image_seq_len", 256),
                max_seq_len=getattr(scheduler.config, "max_image_seq_len", 4096),
                base_shift=getattr(scheduler.config, "base_shift", 0.5),
                max_shift=getattr(scheduler.config, "max_shift", 1.15),
            )
            scheduler.set_timesteps(num_steps, device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_steps, device=device)

        guidance = self._transformer_guidance(bsz, device)

        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            use_grad = grad_last_step_only and (i == len(timesteps) - 1)
            with torch.enable_grad() if use_grad else torch.no_grad():
                packed_in = self.flux2_pipe_cls._pack_latents(latents)
                latent_ids = self.flux2_pipe_cls._prepare_latent_ids(latents).to(device=device, dtype=dtype)

                v_pred = transformer(
                    hidden_states=packed_in.to(dtype),
                    encoder_hidden_states=txt.to(dtype),
                    timestep=(t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0),
                    guidance=guidance,
                    txt_ids=txt_ids.to(dtype),
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
                v_pred = self.flux2_pipe_cls._unpack_latents_with_ids(v_pred, latent_ids)
                latents = scheduler.step(v_pred, t, latents, return_dict=False)[0]

        return latents, z1

    # ── Training step ─────────────────────────────────────────────────────────

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key

        gt_image = self.sample[gt_key].clip(-1, 1)
        bsz = gt_image.shape[0]
        device = self.accelerator.device
        dtype = self.weight_dtype

        if gt_image.shape[1] == 1:
            gt_image = einops.repeat(gt_image, "b 1 h w -> b 3 h w")

        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        lat_spatial_div = vae_scale * 2
        lat_h = gt_image.shape[-2] // lat_spatial_div
        lat_w = gt_image.shape[-1] // lat_spatial_div

        captions = self._pick_captions()
        txt, txt_ids = self._encode_captions(captions, device)

        # ── Phase 1: build a reflow (noise, output) pair with the frozen teacher ──
        with torch.no_grad():
            z0_teacher, z1_teacher = self._rollout(
                self.transformer_fix, txt.to(torch.float16), txt_ids.to(torch.float16),
                lat_h, lat_w, bsz, device, torch.float16, num_steps=self.num_teacher_steps,
                grad_last_step_only=False,
            )
        z0_teacher = z0_teacher.to(dtype)
        z1_teacher = z1_teacher.to(dtype)

        # ── Phase 2: reflow loss — student flow-matching MSE on the straight line ──
        t = torch.rand(bsz, device=device, dtype=dtype).clamp(1e-3, 1 - 1e-3)
        sigma = t.view(bsz, 1, 1, 1)
        x_t = (1 - sigma) * z0_teacher + sigma * z1_teacher
        target_v = z1_teacher - z0_teacher

        guidance = self._transformer_guidance(bsz, device)
        packed_in = self.flux2_pipe_cls._pack_latents(x_t)
        latent_ids = self.flux2_pipe_cls._prepare_latent_ids(x_t).to(device=device, dtype=dtype)

        v_pred = self.transformer(
            hidden_states=packed_in,
            encoder_hidden_states=txt.to(dtype),
            timestep=t,
            guidance=guidance,
            txt_ids=txt_ids.to(dtype),
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        v_pred = self.flux2_pipe_cls._unpack_latents_with_ids(v_pred, latent_ids)
        # The transformer's raw output comes back float32 regardless of the bf16
        # weights/inputs (its final layer upcasts internally for precision), so
        # both operands must be cast explicitly rather than assumed to match.
        loss_reflow = F.mse_loss(v_pred.float(), target_v.float()) * self.lambda_reflow

        # ── Phase 3: pixel-anchor loss — actual K-step student rollout vs real image ──
        x0_student, _ = self._rollout(
            self.transformer, txt, txt_ids, lat_h, lat_w, bsz, device, dtype,
            num_steps=self.num_student_steps, grad_last_step_only=True,
        )
        output_image = decode_latents_flux(x0_student, self.vae, flux2_pipe_cls=self.flux2_pipe_cls).clamp(-1, 1)
        loss_l2 = F.mse_loss(output_image.float(), gt_image.float()) * self.lambda_l2
        loss_lpips = self.net_lpips(output_image.float(), gt_image.float()).mean() * self.lambda_lpips

        loss_gan = output_image.new_tensor(0.0)
        if self.lambda_gan > 0.0:
            pred_fake_g, _ = self._discriminate(output_image, gt_image.detach())
            loss_gan = self.criterionGAN(pred_fake_g, True, for_discriminator=False).mean() * self.lambda_gan

        loss_gen = loss_reflow + loss_l2 + loss_lpips + loss_gan
        self.accelerator.backward(loss_gen)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.gen_params, 0.5)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        # ── Discriminator update ──────────────────────────────────────────────
        loss_disc = self._update_discriminator(output_image.detach(), gt_image)

        return {
            "all": loss_gen + loss_disc,
            "reflow": loss_reflow,
            "l2": loss_l2,
            "lpips": loss_lpips,
            "gan_g": loss_gan,
            "gan_d": loss_disc,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forwardpass(self, captions, height: int = None, width: int = None, num_steps: int = None):
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = len(captions)
        if num_steps is None:
            num_steps = self.num_student_steps
        height = height or self.default_h
        width = width or self.default_w

        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        lat_spatial_div = vae_scale * 2
        lat_h = height // lat_spatial_div
        lat_w = width // lat_spatial_div

        txt, txt_ids = self._encode_captions(captions, device)
        x0_latent, _ = self._rollout(
            self.transformer, txt, txt_ids, lat_h, lat_w, bsz, device, dtype,
            num_steps=num_steps, grad_last_step_only=False,
        )
        return decode_latents_flux(x0_latent, self.vae, flux2_pipe_cls=self.flux2_pipe_cls).clamp(-1, 1)

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()

        dataloader = DataLoader(Subset(self.dataloader.dataset, np.arange(3)), shuffle=False, batch_size=1)
        print(f"Validating on {len(dataloader)} training samples...")
        dataloader = self.accelerator.prepare(dataloader)
        for batch in dataloader:
            idx = self.validate_step(batch, idx, self.dataloader.dataset.gt_key)
        self.accelerator._dataloaders.remove(dataloader)

        print(f"Validating on {len(self.test_dataloader)} test samples...")
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.gt_key)
            if idx > self.max_val_steps:
                break

        gc.collect()
        torch.cuda.empty_cache()
        for model in self.models:
            model.train()

    def validate_step(self, batch, idx, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = self.opt.val.get("num_inference_steps", self.num_student_steps)

        gt = self.sample[gt_key]
        bsz = gt.shape[0]
        captions = self._pick_captions()

        if gt.shape[1] == 1:
            gt = einops.repeat(gt, "b 1 h w -> b 3 h w")

        out = self.forwardpass(captions, height=gt.shape[-2], width=gt.shape[-1], num_steps=num_inference_steps)
        gt_np = (gt.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        for i in range(bsz):
            idx += 1
            images = [gt_np[i], out_np[i]]
            for j in range(len(images)):
                if images[j].shape[0] == 1:
                    images[j] = einops.repeat(images[j], "1 h w -> 3 h w")
            safe_caption = re.sub(r"[^a-zA-Z0-9]+", "_", captions[i][:40]).strip("_")
            log_image(self.opt, self.accelerator, np.hstack(images)[None], f"{idx:04d}_{safe_caption}", self.global_step)
            log_image(self.opt, self.accelerator, out_np[i : i + 1], f"out_{idx:04d}", self.global_step)
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

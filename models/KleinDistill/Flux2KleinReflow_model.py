"""Rectified-Flow (ReFlow) distillation of FLUX.2-klein-4B for image restoration.

Standard flow-matching distillation (see `Flux2KleinDistill_model.py`, VSD-style)
matches the *distribution* of a one-step student against a frozen teacher via an
auxiliary score network. ReFlow instead follows Liu et al., "Flow Straight and
Fast": the frozen teacher is rolled out over several Euler steps from a sampled
noise latent to build a self-consistent (noise, output) pair, and the student is
trained with a plain flow-matching MSE loss on the *straight line* between that
pair. Retraining on straight-line trajectories is what lets the same ODE be
integrated accurately with far fewer discretization steps.

One reflow round straightens trajectories but doesn't perfectly linearize them,
so the student targets a `num_student_steps`-step sampler (4 by default) rather
than a single step — a meaningfully faster inference path than the teacher's
`num_teacher_steps` rollout, without paying the full 1-step error. A pixel loss
(L2 + LPIPS) computed by actually rolling the student out for `num_student_steps`
and comparing to the real GT anchors the straightened trajectory to the true
restoration target (the teacher's own reflow pairs alone could drift from GT).
An optional conditional GAN loss adds high-frequency detail.

Set lambda_gan: 0.0 to disable the discriminator branch.
"""

import gc

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
    prepare_image_latents,
)

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2

try:
    from models.LucidFlux.src.flux.lucidflux import load_precomputed_embeddings
except ImportError:
    load_precomputed_embeddings = None


class Flux2KleinReflow_model(BaseModel):
    """ReFlow distillation of FLUX.2-klein-4B into a `num_student_steps`-step student.

    Components:
      transformer     – student with LoRA (trainable, optimizers[0])
      transformer_fix – frozen teacher (fp16), used only under no_grad to build
                        reflow (noise, output) pairs
      vae             – frozen encoder/decoder
      net_lpips       – frozen perceptual loss
      net_d           – multiscale discriminator (optimizers[1])
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
        self.num_student_steps = int(opt.train.get("num_student_steps", 4))
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

        # ── Discriminator (conditional multiscale PatchGAN) ───────────────────
        self.net_d = MultiscaleDiscriminator()
        self.criterionGAN = GANLoss("hinge")

        # ── Precomputed null-text embeddings ──────────────────────────────────
        embeddings_path = opt.get("precomputed_embeddings_path", None)
        if embeddings_path and load_precomputed_embeddings is not None:
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_txt_ids = embeds["txt_ids"]
        else:
            self.register_buffer_txt = None
            self.register_buffer_txt_ids = None

        # ── Move to device ────────────────────────────────────────────────────
        device = self.accelerator.device
        self.transformer = self.transformer.to(device, dtype=weight_dtype)
        self.transformer_fix = self.transformer_fix.to(device, dtype=torch.float16)
        self.vae = self.vae.to(device, dtype=weight_dtype)
        self.net_lpips = self.net_lpips.to(device)
        self.net_d = self.net_d.to(device)

        # student and discriminator are tracked for checkpointing.
        self.models.append(self.transformer)
        self.models.append(self.net_d)

        total = sum(p.numel() for p in self.transformer.parameters())
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        logger.info(f"{self.transformer.__class__.__name__}: trainable={trainable:,} / total={total:,} ({100*trainable/total:.3f}%)")
        logger.info(f"ReFlow distillation: teacher steps={self.num_teacher_steps}, student steps={self.num_student_steps}")

    # ── Null-text embeddings ──────────────────────────────────────────────────

    def _ensure_null_embeddings(self):
        if self.register_buffer_txt is not None and self.register_buffer_txt_ids is not None:
            return
        self.logger.info("Generating null-text embeddings from Klein text encoders")
        pipe = self.flux2_pipe_cls.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=None, vae=None,
            torch_dtype=self.weight_dtype,
        ).to(self.accelerator.device)
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt([""], num_images_per_prompt=1)
        self.register_buffer_txt = prompt_embeds.cpu()
        self.register_buffer_txt_ids = text_ids.cpu()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def _prepare_text_inputs(self, batch_size: int, device):
        self._ensure_null_embeddings()
        txt = self.register_buffer_txt.to(device).expand(batch_size, -1, -1).to(self.weight_dtype)
        txt_ids = self.register_buffer_txt_ids.to(device).to(self.weight_dtype)
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return txt, txt_ids

    def _transformer_guidance(self, batch_size: int, device):
        if not self.use_transformer_guidance:
            return None
        return torch.full((batch_size,), self.guidance_scale, device=device, dtype=self.weight_dtype)

    # ── Discriminator helpers ─────────────────────────────────────────────────

    def _discriminate(self, lq_image: torch.Tensor, fake_image: torch.Tensor, real_image: torch.Tensor):
        """Run conditional discriminator on fake and real images jointly.

        Both fake and real pass through in the same batch so BatchNorm sees
        consistent statistics (same pattern as LD_model.discriminate).
        """
        fake_input = torch.cat([lq_image, fake_image], dim=1)
        real_input = torch.cat([lq_image, real_image], dim=1)
        fake_and_real = torch.cat([fake_input, real_input], dim=0)
        disc_out = self.net_d(fake_and_real)
        pred_fake, pred_real = [], []
        for scale_out in disc_out:
            pred_fake.append([t[: t.size(0) // 2] for t in scale_out])
            pred_real.append([t[t.size(0) // 2 :] for t in scale_out])
        return pred_fake, pred_real

    def _update_discriminator(self, lq_image: torch.Tensor, fake_image: torch.Tensor, real_image: torch.Tensor):
        """One discriminator update step. fake_image must already be detached."""
        if self.lambda_gan == 0.0:
            return fake_image.new_tensor(0.0)
        pred_fake_d, pred_real_d = self._discriminate(lq_image, fake_image, real_image)
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
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    # ── Shared K-step Euler rollout ───────────────────────────────────────────

    def _rollout(self, transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
                 num_steps, grad_last_step_only=False):
        """Run `num_steps` Euler steps from pure noise through `transformer`, LQ-conditioned.

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

        txt, txt_ids = self._prepare_text_inputs(bsz, device)
        guidance = self._transformer_guidance(bsz, device)

        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            use_grad = grad_last_step_only and (i == len(timesteps) - 1)
            with torch.enable_grad() if use_grad else torch.no_grad():
                packed_in = self.flux2_pipe_cls._pack_latents(latents)
                latent_ids = self.flux2_pipe_cls._prepare_latent_ids(latents).to(device=device, dtype=dtype)
                latent_token_count = packed_in.shape[1]
                model_input = torch.cat([packed_in, image_latents.to(dtype)], dim=1)
                full_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                v_pred = transformer(
                    hidden_states=model_input.to(dtype),
                    encoder_hidden_states=txt.to(dtype),
                    timestep=(t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0),
                    guidance=guidance,
                    txt_ids=txt_ids.to(dtype),
                    img_ids=full_ids.to(dtype),
                    return_dict=False,
                )[0]
                v_pred = v_pred[:, :latent_token_count]
                v_pred = self.flux2_pipe_cls._unpack_latents_with_ids(v_pred, latent_ids)
                latents = scheduler.step(v_pred, t, latents, return_dict=False)[0]

        return latents, z1

    # ── Training step ─────────────────────────────────────────────────────────

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        gt_image = self.sample[gt_key].clip(-1, 1)
        lq_image = self.sample[lq_key].clip(-1, 1)
        bsz = gt_image.shape[0]
        device = self.accelerator.device
        dtype = self.weight_dtype

        if gt_image.shape[1] == 1:
            gt_image = einops.repeat(gt_image, "b 1 h w -> b 3 h w")
        if lq_image.shape[1] == 1:
            lq_image = einops.repeat(lq_image, "b 1 h w -> b 3 h w")

        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        lat_spatial_div = vae_scale * 2
        lat_h = lq_image.shape[-2] // lat_spatial_div
        lat_w = lq_image.shape[-1] // lat_spatial_div

        with torch.no_grad():
            image_latents, image_latent_ids = prepare_image_latents(
                vae=self.vae,
                ref_images=lq_image.to(dtype),
                batch_size=bsz,
                generator=None,
                device=device,
                dtype=dtype,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )

            # ── Phase 1: build a reflow (noise, output) pair with the frozen teacher ──
            z0_teacher, z1_teacher = self._rollout(
                self.transformer_fix, image_latents, image_latent_ids, lat_h, lat_w, bsz, device,
                torch.float16, num_steps=self.num_teacher_steps, grad_last_step_only=False,
            )
            z0_teacher = z0_teacher.to(dtype)
            z1_teacher = z1_teacher.to(dtype)

        # ── Phase 2: reflow loss — student flow-matching MSE on the straight line ──
        t = torch.rand(bsz, device=device, dtype=dtype).clamp(1e-3, 1 - 1e-3)
        sigma = t.view(bsz, 1, 1, 1)
        x_t = (1 - sigma) * z0_teacher + sigma * z1_teacher
        target_v = z1_teacher - z0_teacher

        txt, txt_ids = self._prepare_text_inputs(bsz, device)
        guidance = self._transformer_guidance(bsz, device)
        packed_in = self.flux2_pipe_cls._pack_latents(x_t)
        latent_ids = self.flux2_pipe_cls._prepare_latent_ids(x_t).to(device=device, dtype=dtype)
        latent_token_count = packed_in.shape[1]
        model_input = torch.cat([packed_in, image_latents.to(dtype)], dim=1)
        full_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        v_pred = self.transformer(
            hidden_states=model_input,
            encoder_hidden_states=txt.to(dtype),
            timestep=t,
            guidance=guidance,
            txt_ids=txt_ids.to(dtype),
            img_ids=full_ids.to(dtype),
            return_dict=False,
        )[0]
        v_pred = v_pred[:, :latent_token_count]
        v_pred = self.flux2_pipe_cls._unpack_latents_with_ids(v_pred, latent_ids)
        # The transformer's raw output comes back float32 regardless of the bf16
        # weights/inputs (its final layer upcasts internally for precision), so
        # both operands must be cast explicitly rather than assumed to match.
        loss_reflow = F.mse_loss(v_pred.float(), target_v.float()) * self.lambda_reflow

        # ── Phase 3: pixel-anchor loss — actual K-step student rollout vs real GT ──
        x0_student, _ = self._rollout(
            self.transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
            num_steps=self.num_student_steps, grad_last_step_only=True,
        )
        output_image = decode_latents_flux(x0_student, self.vae, flux2_pipe_cls=self.flux2_pipe_cls).clamp(-1, 1)
        loss_l2 = F.mse_loss(output_image.float(), gt_image.float()) * self.lambda_l2
        loss_lpips = self.net_lpips(output_image.float(), gt_image.float()).mean() * self.lambda_lpips

        loss_gan = output_image.new_tensor(0.0)
        if self.lambda_gan > 0.0:
            pred_fake_g, _ = self._discriminate(lq_image, output_image, gt_image.detach())
            loss_gan = self.criterionGAN(pred_fake_g, True, for_discriminator=False).mean() * self.lambda_gan

        loss_gen = loss_reflow + loss_l2 + loss_lpips + loss_gan
        self.accelerator.backward(loss_gen)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.gen_params, 0.5)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        # ── Discriminator update ──────────────────────────────────────────────
        loss_disc = self._update_discriminator(lq_image, output_image.detach(), gt_image)

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
    def forwardpass(self, lq_image: torch.Tensor, num_steps: int = None):
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = lq_image.shape[0]
        if num_steps is None:
            num_steps = self.num_student_steps

        if lq_image.shape[1] == 1:
            lq_image = einops.repeat(lq_image, "b 1 h w -> b 3 h w")

        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        lat_spatial_div = vae_scale * 2
        lat_h = lq_image.shape[-2] // lat_spatial_div
        lat_w = lq_image.shape[-1] // lat_spatial_div

        image_latents, image_latent_ids = prepare_image_latents(
            vae=self.vae,
            ref_images=lq_image.to(dtype),
            batch_size=bsz,
            generator=None,
            device=device,
            dtype=dtype,
            flux2_pipe_cls=self.flux2_pipe_cls,
        )

        x0_latent, _ = self._rollout(
            self.transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
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
            idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        self.accelerator._dataloaders.remove(dataloader)

        print(f"Validating on {len(self.test_dataloader)} test samples...")
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            if idx > self.max_val_steps:
                break

        gc.collect()
        torch.cuda.empty_cache()
        for model in self.models:
            model.train()

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = self.opt.val.get("num_inference_steps", self.num_student_steps)

        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        bsz = lq.shape[0]

        if lq.shape[1] == 1:
            lq = einops.repeat(lq, "b 1 h w -> b 3 h w")

        out = self.forwardpass(lq, num_steps=num_inference_steps)
        lq_np = (lq.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        gt_np = (gt.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        for i in range(bsz):
            idx += 1
            images = [lq_np[i], gt_np[i], out_np[i]]
            for j in range(len(images)):
                if images[j].shape[0] == 1:
                    images[j] = einops.repeat(images[j], "1 h w -> 3 h w")
            log_image(self.opt, self.accelerator, np.hstack(images)[None], f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator, out_np[i : i + 1], f"out_{idx:04d}", self.global_step)
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

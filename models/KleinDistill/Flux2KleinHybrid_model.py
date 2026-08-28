"""Hybrid trajectory + adversarial + perceptual distillation of FLUX.2-klein-4B
for image restoration.

Motivation: three independently-tuned K=4 recipes in Flux2KleinDistill_model.py
(DINOv2-adversarial, DINOv2-fidelity-only, V-JEPA-conditioned) all converged to
the same 24.9-25.8dB band on Deblur_MIOR's val split (see
playground/eval_klein_checkpoints.py) — swapping the perceptual/adversarial
feature encoder stopped moving the needle. Flux2KleinReflow_model.py's frozen-
teacher rectified-flow consistency loss (Liu et al., "Flow Straight and Fast")
is a genuinely different training signal that none of those three ever used:
it retrains the *trajectory* itself (straight-lining the teacher's own
multi-step ODE solution) rather than only regressing the K-step student's
final output against GT. This model runs both signals together — the same
combination real few-step distillation recipes use (e.g. ADD/LADD, Hyper-SD):
trajectory/consistency distillation + adversarial + perceptual, rather than
any one of those alone.

Per step:
  1. Frozen fp16 teacher (`transformer_fix`) rolls out `num_teacher_steps`
     Euler steps from noise (no_grad) to build a self-consistent (z0, z1)
     pair — Flux2KleinReflow_model.py's Phase 1.
  2. Reflow loss: flow-matching MSE from the *student* at a random point on
     the straight line between that pair (Flux2KleinReflow_model.py's Phase 2).
  3. Pixel-anchor branch: the student's own `num_student_steps`-step rollout
     (gradient kept on the last step only) decoded and compared against real
     GT — L2 + LPIPS + a DINOv2/V-JEPA-2.1 feature-similarity loss, plus a
     pixel-space GAN and a feature-space GAN (Flux2KleinDistill_model.py's
     full recipe, including its separately-tuned discriminator learning rate
     to avoid the discriminator-overpowers-generator collapse documented
     there).

Same cold-start hazard as Flux2KleinDistill_model.py applies: with
grad_last_step_only=True, a cold LoRA's K=4 anchor rollout collapses (see
that file's docstring). Warm up at num_student_steps: 1 before switching to
K=4 in a follow-up config that resumes from the warm-up checkpoint.

Set lambda_reflow: 0 to disable the trajectory-distillation branch (reduces
to Flux2KleinDistill's recipe plus an idle frozen teacher); lambda_gan /
lambda_gan_dino: 0 to disable either GAN branch; lambda_dino: 0 to disable
the feature-similarity term.
"""

import functools
import gc

import einops
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import create_dataset
from models.base_model import BaseModel
from models.archs.related.LD.MultiscaleDiscriminator_arch import MultiscaleDiscriminator
from models.archs.related.LD.loss import GANLoss
from models.reFlux.dino_helpers import extract_dino_tokens, get_dino_feature_dim, load_dinov2_model
from models.reFlux.cross_attn_projector import build_flux_patch_projector, resolve_vjepa_projector_slots
from models.LucidFlux.jepa_helpers import (
    extract_vjepa_tokens,
    get_vjepa_feature_dim,
    load_vjepa2_model,
    load_vjepa2_weights,
    vjepa_from_unit_tensor,
)
from utils import log_image, log_metrics
from utils.dataset_utils import merge_patches, random_aspect_crop_collate

from .dino_discriminator import DinoPatchDiscriminator
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


class Flux2KleinHybrid_model(BaseModel):
    """ReFlow trajectory-consistency + GT pixel/DINO/dual-GAN distillation of FLUX.2-klein-4B.

    Components:
      transformer     – student with LoRA (trainable, optimizers[0])
      transformer_fix – frozen teacher (fp16), used only under no_grad to build
                        reflow (noise, output) pairs
      vae             – frozen encoder/decoder
      net_lpips       – frozen perceptual loss
      dino            – frozen feature encoder (DINOv2 or V-JEPA 2.1, see
                        feature_encoder), feeds the feature-similarity loss
                        and the net_d_dino discriminator
      net_d           – pixel-space multiscale discriminator (trainable, optimizers[1])
      net_d_dino      – DINO/V-JEPA-feature-space patch discriminator (trainable, optimizers[2])
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
        # keys instead of raising AttributeError, so getattr(opt, key, default)
        # never falls back — use .get() everywhere in this file.
        self.lambda_reflow = float(opt.train.get("lambda_reflow", 1.0))
        self.lambda_l2 = float(opt.train.get("lambda_l2", 1.0))
        self.lambda_lpips = float(opt.train.get("lambda_lpips", 1.0))
        self.lambda_dino = float(opt.train.get("lambda_dino", 1.0))
        self.lambda_gan = float(opt.train.get("lambda_gan", 0.02))
        self.lambda_gan_dino = float(opt.train.get("lambda_gan_dino", 0.02))
        self.num_teacher_steps = int(opt.train.get("num_teacher_steps", 8))
        # Number of Euler steps the student rolls out for the pixel-anchor
        # branch, both in training (gradient kept on the last step only, see
        # _rollout) and at inference. 1 recovers one-step distillation and is
        # the required cold-start warm-up value — see module docstring.
        self.num_student_steps = int(opt.train.get("num_student_steps", 1))
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

        # ── Perceptual losses ─────────────────────────────────────────────────
        self.net_lpips = lpips.LPIPS(net="vgg")
        self.net_lpips.requires_grad_(False)

        # Frozen feature encoder behind loss_dino / net_d_dino — see
        # Flux2KleinDistill_model.py, same mechanism ported here unchanged.
        self.feature_encoder = str(opt.get("feature_encoder", "dinov2")).lower()
        if self.feature_encoder == "vjepa2":
            self.dino_model_name = opt.get("vjepa_hub_entry") or "vjepa2_1_vit_large_384"
            self.dino_image_size = int(opt.get("vjepa_image_size") or 384)
            self.dino_dtype = opt.get("vjepa_dtype") or torch.float32
            self.dino_feature_dim = get_vjepa_feature_dim(self.dino_model_name, fallback_dim=opt.get("vjepa_feature_dim") or 1024)
            logger.info(f"Loading V-JEPA 2.1 encoder from torch.hub entry {self.dino_model_name}")
            self.dino = load_vjepa2_model(
                self.dino_model_name, self.accelerator.device, self.dino_dtype,
                pretrained=bool(opt.get("vjepa_pretrained", True)),
            )
            vjepa_checkpoint_path = opt.get("vjepa_checkpoint_path", None)
            if vjepa_checkpoint_path:
                logger.info(f"Loading V-JEPA checkpoint from {vjepa_checkpoint_path}")
                load_vjepa2_weights(self.dino, vjepa_checkpoint_path)
            self.dino.requires_grad_(False)
            self.dino.eval()
        elif self.feature_encoder == "dinov2":
            self.dino_model_name = opt.get("dino_model_name") or "facebook/dinov2-large"
            self.dino_image_size = int(opt.get("dino_image_size") or 518)
            self.dino_dtype = opt.get("dino_dtype") or torch.float32
            self.dino_feature_dim = get_dino_feature_dim(self.dino_model_name, fallback_dim=opt.get("dino_feature_dim") or 1024)
            logger.info(f"Loading DINOv2 encoder from {self.dino_model_name}")
            self.dino = load_dinov2_model(self.dino_model_name, self.accelerator.device, self.dino_dtype)
        else:
            raise ValueError(f"Unsupported feature_encoder: {self.feature_encoder!r} (expected 'dinov2' or 'vjepa2')")

        # ── Discriminator (conditional multiscale PatchGAN) ───────────────────
        self.net_d = MultiscaleDiscriminator()
        self.criterionGAN = GANLoss("hinge")

        # ── DINO/V-JEPA-feature-space discriminator (unconditional PatchGAN) ──
        self.net_d_dino = DinoPatchDiscriminator(feature_dim=self.dino_feature_dim)

        # ── Precomputed null-text embeddings ──────────────────────────────────
        embeddings_path = opt.get("precomputed_embeddings_path", None)
        if embeddings_path and load_precomputed_embeddings is not None:
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_txt_ids = embeds["txt_ids"]
        else:
            self.register_buffer_txt = None
            self.register_buffer_txt_ids = None

        # ── V-JEPA conditioning (extra semantic context tokens for the student's
        # pixel-anchor rollout only — see _prepare_conditioning). Independent of
        # feature_encoder's use for loss_dino/net_d_dino above — reuses the same
        # loaded encoder rather than loading V-JEPA twice. The teacher-pair
        # rollout and the reflow loss deliberately stay on plain null-text
        # conditioning (see optimize_parameters) since the frozen teacher never
        # saw V-JEPA condition tokens.
        self.use_vjepa_condition = bool(opt.get("use_vjepa_condition", False))
        if self.use_vjepa_condition:
            if self.feature_encoder != "vjepa2":
                raise ValueError(
                    "use_vjepa_condition requires feature_encoder: vjepa2 "
                    "(it reuses that encoder for conditioning instead of loading a second one)."
                )
            text_in_dim = getattr(self.transformer.config, "joint_attention_dim", None)
            if text_in_dim is None:
                if self.register_buffer_txt is None:
                    raise ValueError(
                        "Can't determine the transformer's text conditioning dim: "
                        "transformer.config has no joint_attention_dim and no "
                        "precomputed_embeddings_path was given to infer it from."
                    )
                text_in_dim = self.register_buffer_txt.shape[-1]
            self.transformer_context_dim = int(text_in_dim)
            vjepa_condition_num_slots = int(
                opt.get("vjepa_condition_num_slots")
                or resolve_vjepa_projector_slots(opt, self.dino_model_name, self.dino_image_size)
            )
            vjepa_projector_hidden = int(opt.get("vjepa_projector_hidden", 2048))
            self.vjepa_projector = build_flux_patch_projector(
                embed_dim=self.dino_feature_dim,
                transformer_context_dim=self.transformer_context_dim,
                num_slots=vjepa_condition_num_slots,
                hidden_dim=vjepa_projector_hidden,
            )
            self.vjepa_projector.requires_grad_(True)
        else:
            self.vjepa_projector = None

        # ── Move to device ────────────────────────────────────────────────────
        device = self.accelerator.device
        self.transformer = self.transformer.to(device, dtype=weight_dtype)
        self.transformer_fix = self.transformer_fix.to(device, dtype=torch.float16)
        self.vae = self.vae.to(device, dtype=weight_dtype)
        self.net_lpips = self.net_lpips.to(device)
        self.net_d = self.net_d.to(device)
        self.net_d_dino = self.net_d_dino.to(device)
        if self.vjepa_projector is not None:
            self.vjepa_projector = self.vjepa_projector.to(device, dtype=weight_dtype)

        # student, both discriminators, and the (optional) V-JEPA conditioning
        # projector are tracked for checkpointing. transformer_fix is frozen
        # and always reloaded fresh from `pretrained`, so it isn't checkpointed.
        self.models.append(self.transformer)
        self.models.append(self.net_d)
        self.models.append(self.net_d_dino)
        if self.vjepa_projector is not None:
            self.models.append(self.vjepa_projector)

        total = sum(p.numel() for p in self.transformer.parameters())
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        logger.info(f"{self.transformer.__class__.__name__}: trainable={trainable:,} / total={total:,} ({100*trainable/total:.3f}%)")
        logger.info(f"Hybrid distillation: teacher steps={self.num_teacher_steps}, student steps={self.num_student_steps}")

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

    def _prepare_conditioning(self, lq_image: torch.Tensor, batch_size: int, device):
        """Null-text embeddings, extended with projected V-JEPA conditioning
        tokens from the LQ image when use_vjepa_condition is set. Used only
        for the student's pixel-anchor rollout — see class docstring."""
        txt, txt_ids = self._prepare_text_inputs(batch_size, device)
        if not self.use_vjepa_condition:
            return txt, txt_ids

        if txt_ids.shape[0] == 1 and batch_size > 1:
            txt_ids = txt_ids.expand(batch_size, -1, -1)

        vjepa_feat = self._extract_dino_patches(lq_image, requires_grad=False)
        cond_txt = self.vjepa_projector(vjepa_feat.to(self.weight_dtype))
        cond_ids = self.flux2_pipe_cls._prepare_text_ids(cond_txt)
        if cond_ids.ndim == 2:
            cond_ids = cond_ids.unsqueeze(0)
        if cond_ids.shape[0] == 1 and batch_size > 1:
            cond_ids = cond_ids.expand(batch_size, -1, -1)
        cond_ids = cond_ids.to(device=txt_ids.device, dtype=txt_ids.dtype)
        cond_ids[:, :, 0] = 1
        txt = torch.cat([txt, cond_txt.to(txt.dtype)], dim=1)
        txt_ids = torch.cat([txt_ids, cond_ids], dim=1)
        return txt, txt_ids

    def _transformer_guidance(self, batch_size: int, device):
        if not self.use_transformer_guidance:
            return None
        return torch.full((batch_size,), self.guidance_scale, device=device, dtype=self.weight_dtype)

    # ── Feature encoder (DINOv2 or V-JEPA 2.1) ─────────────────────────────────

    def _extract_dino_patches(self, image_pm1: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        """Raw (unnormalized) patch tokens [B, N, D] for an image in [-1, 1].
        Shared by the feature-similarity loss, the feature-space discriminator,
        and (when use_vjepa_condition) the conditioning projector."""
        img_01 = (image_pm1 * 0.5 + 0.5).clamp(0, 1)
        with torch.enable_grad() if requires_grad else torch.no_grad():
            pixels = vjepa_from_unit_tensor(img_01, size=self.dino_image_size, device=img_01.device, out_dtype=self.dino_dtype)
            if self.feature_encoder == "vjepa2":
                feat = extract_vjepa_tokens(self.dino, pixels, device=img_01.device, dtype=self.dino_dtype)
            else:
                feat = extract_dino_tokens(self.dino, pixels, device=img_01.device, dtype=self.dino_dtype)
        return feat.float()

    # ── Discriminator helpers ─────────────────────────────────────────────────

    def _discriminate(self, lq_image: torch.Tensor, fake_image: torch.Tensor, real_image: torch.Tensor):
        """Run conditional discriminator on fake and real images jointly."""
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

    def _update_discriminator_dino(self, fake_feat: torch.Tensor, real_feat: torch.Tensor):
        """One feature-space discriminator update step. fake_feat must already
        be detached; real_feat is expected to already be detached too."""
        if self.lambda_gan_dino == 0.0:
            return fake_feat.new_tensor(0.0)
        pred_fake_d = self.net_d_dino(fake_feat)
        pred_real_d = self.net_d_dino(real_feat)
        loss_d_fake = self.criterionGAN(pred_fake_d, False, for_discriminator=True).mean()
        loss_d_real = self.criterionGAN(pred_real_d, True, for_discriminator=True).mean()
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.accelerator.backward(loss_d)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.disc_dino_params, 1.0)
        self.optimizers[2].step()
        self.optimizers[2].zero_grad()
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
        if self.vjepa_projector is not None:
            self.gen_params += list(self.vjepa_projector.parameters())
        self.optimizers.append(optimizer_class(self.gen_params, **common_kwargs))

        # Discriminator LR kept separate from the generator's (defaults to
        # matching it if unset) — see Flux2KleinDistill_model.py's module
        # docstring for the discriminator-overpowers-generator collapse this
        # avoids: a tiny LoRA generator (~0.3% of params) can't out-adapt a
        # freely-trainable discriminator at the same learning rate.
        d_lr = opt.get("d_learning_rate", opt.learning_rate)
        self.disc_params = list(self.net_d.parameters())
        self.optimizers.append(optimizer_class(self.disc_params, lr=d_lr,
                                               betas=(opt.adam_beta1, opt.adam_beta2),
                                               weight_decay=opt.get("adam_weight_decay", 0.01),
                                               eps=opt.get("adam_epsilon", 1e-8)))

        self.disc_dino_params = list(self.net_d_dino.parameters())
        self.optimizers.append(optimizer_class(self.disc_dino_params, lr=d_lr,
                                               betas=(opt.adam_beta1, opt.adam_beta2),
                                               weight_decay=opt.get("adam_weight_decay", 0.01),
                                               eps=opt.get("adam_epsilon", 1e-8)))

    def after_resume_checkpoint(self, load_path):
        """Re-apply this config's generator learning rate after a resume.

        accelerate 1.1.0's `save_state(..., save_optimizer=False)` call in
        base_model.py doesn't actually suppress optimizer/scheduler saving —
        `Accelerator.save_state` in this version has no `save_optimizer`
        parameter, so it's silently absorbed and ignored; optimizer.bin /
        scheduler.bin are written and reloaded on every checkpoint regardless.
        The "constant" scheduler is a `LambdaLR` whose `state_dict()` includes
        `base_lrs`, so `accelerator.load_state()` (called just before this
        hook, see base_model.py's train()) restores the *checkpointed* LR —
        silently overriding any `train.optim.learning_rate` change made in
        this config for a new phase, since the next `scheduler.step()` just
        recomputes `param_groups['lr']` from the reloaded `base_lrs`. Only the
        generator's optimizer/scheduler (index 0) is touched here — the
        discriminator LRs are a deliberately separate, already-tuned
        mechanism (see module docstring) and aren't part of what a "make the
        generator learn faster" config change should affect.

        No-ops outside training: BaseModel.prepare() only calls
        setup_optimizers()/setup_schedulers() `if self.is_train`, so
        self.optimizers/self.schedulers are empty lists for eval-only runs
        (e.g. playground/eval_klein_checkpoints.py, which resumes checkpoints
        with is_train=False) — indexing [0] into them would raise IndexError
        for a resume path this method has no reason to touch anyway.
        """
        if not self.optimizers or not self.schedulers:
            return
        new_lr = float(self.opt.train.optim.learning_rate)
        for group in self.optimizers[0].param_groups:
            group["lr"] = new_lr
        # accelerator.prepare() wraps the raw LambdaLR in an AcceleratedScheduler
        # that holds it as `.scheduler` and does *not* forward attribute access
        # (no __getattr__) — `base_lrs`/`_last_lr` live one level down.
        gen_scheduler = self.schedulers[0]
        inner_scheduler = getattr(gen_scheduler, "scheduler", gen_scheduler)
        if hasattr(inner_scheduler, "base_lrs"):
            inner_scheduler.base_lrs = [new_lr for _ in inner_scheduler.base_lrs]
        if hasattr(inner_scheduler, "_last_lr"):
            inner_scheduler._last_lr = [new_lr for _ in inner_scheduler._last_lr]
        self.logger.info(
            f"after_resume_checkpoint: overrode resumed generator LR to {new_lr} "
            f"from this config (discarding whatever LR checkpoint-{load_path} was trained at)"
        )

    # ── Dataloaders ──────────────────────────────────────────────────────────

    def setup_dataloaders(self):
        """Same as BaseModel.setup_dataloaders, except the train loader crops
        with random_aspect_crop_collate instead of relying on a per-row dataset
        transform — see Flux2KleinDistill.yml's datasets.train comment for why
        that has to happen at collate time rather than in the dataset's cached
        .map() pipeline (source images are 1920x1080)."""
        self.logger.info("Setting up val dataloaders...")
        val_set = create_dataset(self.opt.datasets.val)
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=self.opt.datasets.val.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=self.opt.datasets.val.get('num_worker_per_gpu', 0),
        )

        self.logger.info("Setting up train dataloader...")
        train_set = create_dataset(self.opt.datasets.train)
        collate_fn = functools.partial(
            random_aspect_crop_collate,
            keys=[train_set.lq_key, train_set.gt_key],
            area_h=int(self.opt.train.get("crop_area_h", 512)),
            area_w=int(self.opt.train.get("crop_area_w", 512)),
            scale=tuple(self.opt.train.get("crop_scale", [0.5, 1.0])),
            ratio=tuple(self.opt.train.get("crop_ratio", [0.75, 1.3333])),
        )
        self.dataloader = DataLoader(
            train_set,
            shuffle=self.opt.datasets.train.use_shuffle,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.datasets.train.get('num_worker_per_gpu', 0),
            collate_fn=collate_fn,
        )

    # ── Data feeding ──────────────────────────────────────────────────────────

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    # ── Core forward: multi-step Euler rollout, LQ-conditioned ─────────────────

    def _rollout(self, transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
                 num_steps, grad_last_step_only=False, txt=None, txt_ids=None):
        """Run `num_steps` Euler steps from pure noise through `transformer`, LQ-conditioned.

        Returns (x0_final, z1_initial_noise) — the initial noise is needed by
        the teacher call to build a reflow pair (Liu et al., straight line
        from z1 to x0), and is simply unused by callers that only want x0
        (the student's pixel-anchor rollout, inference).

        By default every step runs under no_grad. When `grad_last_step_only`
        is True, only the final step keeps gradients, so a downstream loss can
        backprop into the student's parameters without holding activations for
        every discretization step in the rollout.

        txt/txt_ids: precomputed conditioning (see _prepare_conditioning); if
        omitted, falls back to plain null-text (_prepare_text_inputs).
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

        if txt is None or txt_ids is None:
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
        loss_reflow = gt_image.new_tensor(0.0)
        if self.lambda_reflow > 0.0:
            t = torch.rand(bsz, device=device, dtype=dtype).clamp(1e-3, 1 - 1e-3)
            sigma = t.view(bsz, 1, 1, 1)
            x_t = (1 - sigma) * z0_teacher + sigma * z1_teacher
            target_v = z1_teacher - z0_teacher

            txt_r, txt_ids_r = self._prepare_text_inputs(bsz, device)
            guidance_r = self._transformer_guidance(bsz, device)
            packed_in = self.flux2_pipe_cls._pack_latents(x_t)
            latent_ids = self.flux2_pipe_cls._prepare_latent_ids(x_t).to(device=device, dtype=dtype)
            latent_token_count = packed_in.shape[1]
            model_input = torch.cat([packed_in, image_latents.to(dtype)], dim=1)
            full_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            v_pred = self.transformer(
                hidden_states=model_input,
                encoder_hidden_states=txt_r.to(dtype),
                timestep=t,
                guidance=guidance_r,
                txt_ids=txt_ids_r.to(dtype),
                img_ids=full_ids.to(dtype),
                return_dict=False,
            )[0]
            v_pred = v_pred[:, :latent_token_count]
            v_pred = self.flux2_pipe_cls._unpack_latents_with_ids(v_pred, latent_ids)
            # The transformer's raw output comes back float32 regardless of the
            # bf16 weights/inputs (its final layer upcasts internally for
            # precision), so both operands must be cast explicitly.
            loss_reflow = F.mse_loss(v_pred.float(), target_v.float()) * self.lambda_reflow

        # ── Phase 3: pixel/DINO/GAN anchor — K-step student rollout vs real GT ──
        txt, txt_ids = self._prepare_conditioning(lq_image, bsz, device)
        x0_student, _ = self._rollout(
            self.transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
            num_steps=self.num_student_steps, grad_last_step_only=True,
            txt=txt, txt_ids=txt_ids,
        )
        output_image = decode_latents_flux(x0_student, self.vae, flux2_pipe_cls=self.flux2_pipe_cls).clamp(-1, 1)

        loss_l2 = F.mse_loss(output_image.float(), gt_image.float()) * self.lambda_l2
        loss_lpips = self.net_lpips(output_image.float(), gt_image.float()).mean() * self.lambda_lpips

        out_dino_feat = self._extract_dino_patches(output_image, requires_grad=True)
        with torch.no_grad():
            gt_dino_feat = self._extract_dino_patches(gt_image, requires_grad=False)
        loss_dino = (1 - (F.normalize(out_dino_feat, dim=-1) * F.normalize(gt_dino_feat, dim=-1)).sum(dim=-1)).mean()
        loss_dino = loss_dino * self.lambda_dino

        loss_gan = output_image.new_tensor(0.0)
        if self.lambda_gan > 0.0:
            pred_fake_g, _ = self._discriminate(lq_image, output_image, gt_image.detach())
            loss_gan = self.criterionGAN(pred_fake_g, True, for_discriminator=False).mean() * self.lambda_gan

        loss_gan_dino = output_image.new_tensor(0.0)
        if self.lambda_gan_dino > 0.0:
            pred_fake_dino_g = self.net_d_dino(out_dino_feat)
            loss_gan_dino = self.criterionGAN(pred_fake_dino_g, True, for_discriminator=False).mean() * self.lambda_gan_dino

        loss_gen = loss_reflow + loss_l2 + loss_lpips + loss_dino + loss_gan + loss_gan_dino
        self.accelerator.backward(loss_gen)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.gen_params, 0.5)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        # ── Discriminator updates ──────────────────────────────────────────────
        loss_disc = self._update_discriminator(lq_image, output_image.detach(), gt_image)
        loss_disc_dino = self._update_discriminator_dino(out_dino_feat.detach(), gt_dino_feat)

        return {
            "all": loss_gen + loss_disc + loss_disc_dino,
            "reflow": loss_reflow,
            "l2": loss_l2,
            "lpips": loss_lpips,
            "dino": loss_dino,
            "gan_g": loss_gan,
            "gan_d": loss_disc,
            "gan_dino_g": loss_gan_dino,
            "gan_dino_d": loss_disc_dino,
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

        txt, txt_ids = self._prepare_conditioning(lq_image, bsz, device)

        x0_latent, _ = self._rollout(
            self.transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
            num_steps=num_steps, grad_last_step_only=False,
            txt=txt, txt_ids=txt_ids,
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

        dataloader = DataLoader(Subset(self.dataloader.dataset, np.arange(3)), shuffle=False, batch_size=1,
                                collate_fn=self.dataloader.collate_fn)
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

        if self.opt.val.patched:
            b, c, h, w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():
                patch = self.sample[lq_key]
                if patch.shape[1] == 1:
                    patch = einops.repeat(patch, "b 1 h w -> b 3 h w")
                pred.append(self.forwardpass(patch, num_steps=num_inference_steps))
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, "(b n) c h w -> b n c h w", b=b)
            out = torch.stack([
                merge_patches(pred[i], self.sample[lq_key + "_patched_pos"])[..., :h, :w]
                for i in range(len(pred))
            ])
            lq = self.sample[lq_key + "_original"]
            gt = self.sample[gt_key + "_original"]
        else:
            lq = self.sample[lq_key]
            gt = self.sample[gt_key]
            if lq.shape[1] == 1:
                lq = einops.repeat(lq, "b 1 h w -> b 3 h w")
            out = self.forwardpass(lq, num_steps=num_inference_steps)

        bsz = gt.shape[0]
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

"""One-step distillation of FLUX.2-klein-4B for image restoration.

Student (LoRA transformer) learns to restore images from LQ reference in a
single denoising step using pixel supervision (L2 + LPIPS), a feature
similarity loss against a frozen encoder (DINOv2 by default, or V-JEPA 2.1
via feature_encoder: vjepa2), a conditional hinge GAN loss with a
multiscale pixel-space patch discriminator, and a second hinge GAN loss with
a patch discriminator operating on that same encoder's features instead of
pixels (cf. vision-aided/projected GAN discriminators — classifying
real/fake in a frozen pretrained feature space tends to give a stronger,
faster-converging adversarial signal than a from-scratch pixel discriminator
alone).

VSD distribution-matching was tried and dropped: it reproducibly collapsed
loss_kl/loss_gan_d to ~0 (output degenerating toward near-black) within a
few thousand steps regardless of GAN loss weighting — the "fake" score
network (a LoRA copy trained on un-conditioned noised student outputs, one
step per generator step) converged to agreeing with the frozen teacher on a
low-diversity mode faster than it tracked the student's real distribution.
Pixel + DINO + GAN is the more stable fallback the module docstring used to
recommend as an ablation; DINO adds a coarser, more object/structure-level
perceptual signal than LPIPS to compensate for losing VSD's realism pressure.

Set lambda_gan: 0 to disable the pixel GAN; lambda_gan_dino: 0 to disable the
DINO-feature GAN; lambda_dino: 0 to disable the DINO perceptual term.
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


class Flux2KleinDistill_model(BaseModel):
    """One-step flow-matching distillation of FLUX.2-klein-4B.

    Components:
      transformer – student with LoRA (trainable, optimizers[0])
      vae         – frozen encoder/decoder
      net_lpips   – frozen perceptual loss
      dino        – frozen feature encoder (DINOv2 or V-JEPA 2.1, see
                    feature_encoder), feeds the feature-similarity loss
                    and the net_d_dino discriminator
      net_d       – pixel-space multiscale discriminator (trainable, optimizers[1])
      net_d_dino  – DINO-feature-space patch discriminator (trainable, optimizers[2])
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
        self.lambda_l2 = float(opt.train.get("lambda_l2", 1.0))
        self.lambda_lpips = float(opt.train.get("lambda_lpips", 1.0))
        self.lambda_dino = float(opt.train.get("lambda_dino", 1.0))
        self.lambda_gan = float(opt.train.get("lambda_gan", 0.1))
        # Kept conservative like lambda_gan (see module docstring re: the
        # earlier discriminator-overpowers-generator collapse with lambda_gan
        # at 0.1 and a matched discriminator/generator learning rate).
        self.lambda_gan_dino = float(opt.train.get("lambda_gan_dino", 0.02))
        # Number of Euler steps the student rolls out, both in training
        # (gradient kept on the last step only, see _rollout) and at
        # inference. 1 recovers the original one-step-distillation behavior.
        self.num_student_steps = int(opt.train.get("num_student_steps", 1))
        lora_rank = int(opt.train.get("lora_rank", 16))
        lora_alpha = int(opt.train.get("lora_alpha", lora_rank * 2))
        lora_dropout = float(opt.train.get("lora_dropout", 0.0))
        lora_target_modules = opt.train.get("lora_target_modules", None)

        # ── VAE ──────────────────────────────────────────────────────────────
        self.vae = AutoencoderKLFlux2.from_pretrained(pretrained, subfolder="vae", revision=revision, variant=variant)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ── Noise scheduler ───────────────────────────────────────────────────
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

        # ── Perceptual losses ─────────────────────────────────────────────────
        self.net_lpips = lpips.LPIPS(net="vgg")
        self.net_lpips.requires_grad_(False)

        # Frozen feature encoder behind loss_dino / net_d_dino. Set
        # feature_encoder: vjepa2 to swap DINOv2 for V-JEPA 2.1 — same role
        # (perceptual similarity target + feature-space discriminator input),
        # different backbone. Attribute names keep the `dino_*` prefix
        # regardless of which encoder is active, to minimize the diff between
        # the two paths (self.dino_feature_dim etc. feed DinoPatchDiscriminator,
        # which is dimension-agnostic).
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
        # Receives [lq_image, fake_or_real] concatenated along channels (6 ch).
        self.net_d = MultiscaleDiscriminator()
        self.criterionGAN = GANLoss("hinge")

        # ── DINO-feature-space discriminator (unconditional PatchGAN) ─────────
        # Not LQ-conditioned, mirroring how the old VSD branch ran unconditioned
        # ("pure generation quality") — it's judging realism in DINO feature
        # space, not fidelity to a specific input, which the pixel discriminator
        # and pixel/LPIPS/DINO-similarity losses already cover.
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

        # ── V-JEPA conditioning (extra semantic context tokens, on top of the
        # existing VAE-latent LQ conditioning already concatenated into the
        # img stream in _rollout) ───────────────────────────────────────────
        # Projects the LQ image's frozen-encoder patch tokens through a
        # learned-query cross-attention projector (same mechanism as
        # ContextFluxBaseline_model.py's jepa_projector, text-head only) into
        # extra txt-sequence tokens, concatenated onto the null-text
        # embeddings the transformer already conditions on. Independent of
        # feature_encoder's use for loss_dino/net_d_dino above — reuses the
        # same loaded encoder rather than loading V-JEPA twice.
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
        self.vae = self.vae.to(device, dtype=weight_dtype)
        self.net_lpips = self.net_lpips.to(device)
        self.net_d = self.net_d.to(device)
        self.net_d_dino = self.net_d_dino.to(device)
        if self.vjepa_projector is not None:
            self.vjepa_projector = self.vjepa_projector.to(device, dtype=weight_dtype)

        # student, both discriminators, and the (optional) V-JEPA conditioning
        # projector are tracked for checkpointing.
        self.models.append(self.transformer)
        self.models.append(self.net_d)
        self.models.append(self.net_d_dino)
        if self.vjepa_projector is not None:
            self.models.append(self.vjepa_projector)

        total = sum(p.numel() for p in self.transformer.parameters())
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        logger.info(f"{self.transformer.__class__.__name__}: trainable={trainable:,} / total={total:,} ({100*trainable/total:.3f}%)")

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
        tokens from the LQ image when use_vjepa_condition is set.

        Computed once per forward pass and threaded through _rollout, rather
        than recomputed per denoising step — the conditioning doesn't change
        across steps. The V-JEPA encoder itself stays frozen/no_grad (it's
        read-only conditioning here, not a backprop target like
        _extract_dino_patches for loss_dino); only self.vjepa_projector
        trains.
        """
        txt, txt_ids = self._prepare_text_inputs(batch_size, device)
        if not self.use_vjepa_condition:
            return txt, txt_ids

        # _prepare_text_inputs only broadcasts txt_ids to batch_size when it
        # started 2D; precomputed embeddings can already be 3D with a leading
        # dim of 1, which _rollout tolerates on its own (txt_ids never gets
        # concatenated with anything batch-expanded there) but which breaks
        # the torch.cat below once cond_ids (always expanded to batch_size)
        # joins it.
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
        # Flags these as conditioning tokens distinct from the null-text
        # sequence's own position ids, mirroring
        # ContextFluxBaseline_model.py's _concat_text_conditioning.
        cond_ids[:, :, 0] = 1

        txt = torch.cat([txt, cond_txt.to(device=txt.device, dtype=txt.dtype)], dim=1)
        txt_ids = torch.cat([txt_ids, cond_ids], dim=1)
        return txt, txt_ids

    def _transformer_guidance(self, batch_size: int, device):
        if not self.use_transformer_guidance:
            return None
        return torch.full((batch_size,), self.guidance_scale, device=device, dtype=self.weight_dtype)

    # ── Feature encoder (DINOv2 or V-JEPA 2.1) ─────────────────────────────────

    def _extract_dino_patches(self, image_pm1: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        """Raw (unnormalized) patch tokens [B, N, D] for an image in [-1, 1],
        from whichever encoder feature_encoder selects (DINOv2 or V-JEPA 2.1).

        Shared by the feature-similarity loss (cosine distance on
        L2-normalized features) and the feature-space discriminator (raw
        features, so it can also use magnitude, not just direction).
        `self.dino` is frozen (requires_grad_(False) at load time), so no
        gradient accumulates into it — this only needs the forward pass to
        stay differentiable w.r.t. the input for the student's own output
        (requires_grad=True), exactly like `self.net_lpips` above is called
        without a no_grad wrapper.
        """
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
        """Run conditional discriminator on fake and real images jointly.

        Both fake and real pass through in the same batch so BatchNorm sees
        consistent statistics (same pattern as LD_model.discriminate).
        """
        fake_input = torch.cat([lq_image, fake_image], dim=1)
        real_input = torch.cat([lq_image, real_image], dim=1)
        fake_and_real = torch.cat([fake_input, real_input], dim=0)
        disc_out = self.net_d(fake_and_real)
        # Split back into fake / real halves.
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
        """One DINO-feature discriminator update step. fake_feat must already
        be detached; real_feat is expected to already be detached too (it
        should come from an _extract_dino_patches(..., requires_grad=False)
        call)."""
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

        self.disc_params = list(self.net_d.parameters())
        d_lr = opt.get("d_learning_rate", opt.learning_rate)
        self.optimizers.append(optimizer_class(self.disc_params, lr=d_lr,
                                               betas=(opt.adam_beta1, opt.adam_beta2),
                                               weight_decay=opt.get("adam_weight_decay", 0.01),
                                               eps=opt.get("adam_epsilon", 1e-8)))

        self.disc_dino_params = list(self.net_d_dino.parameters())
        self.optimizers.append(optimizer_class(self.disc_dino_params, lr=d_lr,
                                               betas=(opt.adam_beta1, opt.adam_beta2),
                                               weight_decay=opt.get("adam_weight_decay", 0.01),
                                               eps=opt.get("adam_epsilon", 1e-8)))

    # ── Dataloaders ──────────────────────────────────────────────────────────

    def setup_dataloaders(self):
        """Same as BaseModel.setup_dataloaders, except the train loader crops
        with random_aspect_crop_collate instead of relying on a per-row dataset
        transform — see the comment on datasets.train.transforms in the yml for
        why that has to happen at collate time rather than in the dataset's
        cached .map() pipeline.
        """
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

    # ── Core forward: one-step student denoise ────────────────────────────────

    def _rollout(self, transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
                 num_steps, grad_last_step_only=False, txt=None, txt_ids=None):
        """Run `num_steps` Euler steps from pure noise through `transformer`, LQ-conditioned.

        By default every step runs under no_grad (used for inference). When
        `grad_last_step_only` is True, only the final step keeps gradients, so
        training on a multi-step rollout doesn't need to hold activations for
        every discretization step — same trick as Flux2KleinReflow_model.py's
        _rollout. With num_steps=1 this reduces to exactly the old
        _one_step_denoise behavior (the single step is trivially the last one).

        txt/txt_ids: precomputed conditioning (see _prepare_conditioning); if
        omitted, falls back to plain null-text (_prepare_text_inputs) with no
        V-JEPA conditioning, regardless of use_vjepa_condition.
        """
        lat_c = transformer.config.in_channels
        latents = torch.randn(bsz, lat_c, lat_h, lat_w, device=device, dtype=dtype)

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

        return latents

    # ── Distillation loss ─────────────────────────────────────────────────────

    def _optimize_parameters(self, denoised_latents, gt_image, lq_image, image_latents, image_latent_ids):
        output_image = decode_latents_flux(denoised_latents, self.vae, flux2_pipe_cls=self.flux2_pipe_cls).clamp(-1, 1)

        loss_l2 = F.mse_loss(output_image.float(), gt_image.float()) * self.lambda_l2
        loss_lpips = self.net_lpips(output_image.float(), gt_image.float()).mean() * self.lambda_lpips

        # DINO features, shared by the perceptual loss and the DINO-space discriminator.
        out_dino_feat = self._extract_dino_patches(output_image, requires_grad=True)
        with torch.no_grad():
            gt_dino_feat = self._extract_dino_patches(gt_image, requires_grad=False)
        loss_dino = (1 - (F.normalize(out_dino_feat, dim=-1) * F.normalize(gt_dino_feat, dim=-1)).sum(dim=-1)).mean()
        loss_dino = loss_dino * self.lambda_dino

        # ── Pixel-space GAN generator loss (fool the pixel discriminator) ─────
        loss_gan = output_image.new_tensor(0.0)
        if self.lambda_gan > 0.0:
            pred_fake_g, _ = self._discriminate(lq_image, output_image, gt_image.detach())
            loss_gan = self.criterionGAN(pred_fake_g, True, for_discriminator=False).mean() * self.lambda_gan

        # ── DINO-space GAN generator loss (fool the DINO-feature discriminator) ──
        loss_gan_dino = output_image.new_tensor(0.0)
        if self.lambda_gan_dino > 0.0:
            pred_fake_dino_g = self.net_d_dino(out_dino_feat)
            loss_gan_dino = self.criterionGAN(pred_fake_dino_g, True, for_discriminator=False).mean() * self.lambda_gan_dino

        loss_gen = loss_l2 + loss_lpips + loss_dino + loss_gan + loss_gan_dino
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
            "l2": loss_l2,
            "lpips": loss_lpips,
            "dino": loss_dino,
            "gan_g": loss_gan,
            "gan_d": loss_disc,
            "gan_dino_g": loss_gan_dino,
            "gan_dino_d": loss_disc_dino,
        }

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

        # Compute spatial dimensions from LQ.
        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        lat_spatial_div = vae_scale * 2
        lat_h = lq_image.shape[-2] // lat_spatial_div
        lat_w = lq_image.shape[-1] // lat_spatial_div

        with torch.inference_mode():
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

        # K-step student rollout (K=1 recovers the original one-step behavior).
        # Gradient is kept only on the last step so training doesn't need to
        # hold activations for every discretization step.
        x0_latent = self._rollout(
            self.transformer, image_latents, image_latent_ids, lat_h, lat_w, bsz, device, dtype,
            num_steps=self.num_student_steps, grad_last_step_only=True,
            txt=txt, txt_ids=txt_ids,
        )

        return self._optimize_parameters(x0_latent, gt_image, lq_image, image_latents, image_latent_ids)

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

        x0_latent = self._rollout(
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
            # Full-resolution inference: feed_data's grids() call (triggered by
            # val.patched) already split lq/gt into overlapping patches and set
            # self.sample[lq_key] = None; self.original_size holds the
            # pre-patch shape. Run each minibatch of patches through the same
            # forwardpass() used for the non-patched case, then stitch the
            # per-patch outputs back into one image with merge_patches
            # (Bartlett-window overlap blending) — same pattern as
            # OSEDiff_onedataset_model.py's validate_step.
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

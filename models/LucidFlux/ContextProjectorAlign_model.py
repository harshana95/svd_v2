import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Flux2Transformer2DModel
try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2

from models.base_model import BaseModel
from utils import log_image, log_metrics
from utils.misc import log_metric
from utils.loss import Loss

from .clip_helpers import (
    clip_from_unit_tensor,
    extract_clip_tokens,
    get_clip_feature_dim,
    get_clip_num_patches,
    load_clip_model,
)
from .cross_attn_projector import PatchCrossAttnProjector
from .dino_helpers import (
    extract_dino_tokens,
    get_dino_feature_dim,
    load_dinov2_model,
)
from .helpers import (
    decode_packed_vae_tokens,
    encode_vae_packed_tokens,
    get_flux2_pipeline_class,
    get_latent_token_layout,
)
from .jepa_helpers import (
    extract_vjepa_tokens,
    get_vjepa_feature_dim,
    load_vjepa2_model,
    load_vjepa2_weights,
    vjepa_from_unit_tensor,
)
from .vljepa_helpers import (
    VLJEPA_EMBED_DIM,
    load_open_vljepa,
    load_vljepa_tokenizers,
    predict_semantic_embedding,
    preprocess_vljepa_image,
    tokenize_vljepa_query,
)
from .src.flux.lucidflux import load_swinir


def _cfg(opt, key, default=None):
    """Read config; DictAsMember returns None for missing keys (not the default)."""
    value = opt.get(key, default) if hasattr(opt, "get") else getattr(opt, key, default)
    if value in (None, "~"):
        return default
    return value


def _cfg_bool(opt, key, default=False):
    value = _cfg(opt, key, default)
    if isinstance(value, bool):
        return value
    return default if value in (None, "~", "") else bool(value)


def _cfg_int(opt, key, default):
    return int(_cfg(opt, key, default))


class ContextProjectorAlign_model(BaseModel):
    """Train a projector to map context-extractor tokens to Flux VAE packed tokens."""

    def __init__(self, opt, logger):
        super(ContextProjectorAlign_model, self).__init__(opt, logger)

        self.use_vljepa_condition = _cfg_bool(opt, "use_vljepa_condition", False)
        self.use_vjepa_condition = _cfg_bool(opt, "use_vjepa_condition", True)
        self.use_dino_condition = _cfg_bool(opt, "use_dino_condition", False)
        self.use_clip_condition = _cfg_bool(opt, "use_clip_condition", False)
        if self.use_vljepa_condition:
            if self.use_vjepa_condition:
                logger.warning(
                    "Both use_vljepa_condition and use_vjepa_condition are enabled; "
                    "preferring VL-JEPA and disabling V-JEPA."
                )
                self.use_vjepa_condition = False
            if self.use_dino_condition:
                logger.warning(
                    "Both use_vljepa_condition and use_dino_condition are enabled; "
                    "preferring VL-JEPA and disabling DINO."
                )
                self.use_dino_condition = False
            if self.use_clip_condition:
                logger.warning(
                    "Both use_vljepa_condition and use_clip_condition are enabled; "
                    "preferring VL-JEPA and disabling CLIP."
                )
                self.use_clip_condition = False
        elif self.use_vjepa_condition:
            if self.use_dino_condition:
                logger.warning(
                    "Both use_vjepa_condition and use_dino_condition are enabled; "
                    "preferring V-JEPA and disabling DINO."
                )
                self.use_dino_condition = False
            if self.use_clip_condition:
                logger.warning(
                    "Both use_vjepa_condition and use_clip_condition are enabled; "
                    "preferring V-JEPA and disabling CLIP."
                )
                self.use_clip_condition = False
        elif self.use_dino_condition and self.use_clip_condition:
            logger.warning(
                "Both use_dino_condition and use_clip_condition are enabled; "
                "preferring DINO and disabling CLIP."
            )
            self.use_clip_condition = False

        if not (
            self.use_vljepa_condition
            or self.use_vjepa_condition
            or self.use_dino_condition
            or self.use_clip_condition
        ):
            raise ValueError(
                "ContextProjectorAlign_model requires one of "
                "use_vljepa_condition, use_vjepa_condition, use_dino_condition, or use_clip_condition."
            )

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        self.vjepa_dtype = _cfg(opt, "vjepa_dtype", torch.float32)

        pretrained = opt.pretrained_model_name_or_path
        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)
        transformer_config = Flux2Transformer2DModel.load_config(
            pretrained, subfolder="transformer"
        )

        self.source_image = _cfg(opt, "source_image", "gt")
        self.train_crop_size = _cfg_int(opt, "train_crop_size", 512)
        self.decode_tokens = _cfg_bool(opt.val, "decode_tokens", True)

        self.vjepa_hub_entry = _cfg(opt, "vjepa_hub_entry", "vjepa2_1_vit_large_384")
        self.vjepa_image_size = _cfg_int(opt, "vjepa_image_size", 384)
        self.vjepa_pretrained = _cfg_bool(opt, "vjepa_pretrained", True)
        self.vjepa_checkpoint_path = _cfg(opt, "vjepa_checkpoint_path", None)
        self.vjepa_feature_dim = get_vjepa_feature_dim(
            self.vjepa_hub_entry,
            fallback_dim=_cfg_int(opt, "vjepa_feature_dim", 1024),
        )

        self.dino_model_name = None
        self.dino_image_size = None
        self.dino_dtype = None
        self.dino_feature_dim = None
        if self.use_dino_condition:
            self.dino_model_name = opt.get("dino_model_name") or "facebook/dinov2-large"
            self.dino_image_size = opt.get("dino_image_size") or 518
            self.dino_dtype = opt.get("dino_dtype") or torch.float32
            self.dino_feature_dim = get_dino_feature_dim(
                self.dino_model_name,
                fallback_dim=opt.get("dino_feature_dim") or 1024,
            )

        self.clip_model_name = None
        self.clip_image_size = None
        self.clip_dtype = None
        self.clip_feature_dim = None
        if self.use_clip_condition:
            self.clip_model_name = opt.get("clip_model_name") or "openai/clip-vit-large-patch14"
            self.clip_image_size = opt.get("clip_image_size") or 224
            self.clip_dtype = opt.get("clip_dtype") or torch.float32
            self.clip_feature_dim = get_clip_feature_dim(
                self.clip_model_name,
                fallback_dim=opt.get("clip_feature_dim") or 1024,
            )

        self.vljepa_model = None
        self.vljepa_query_tokenizer = None
        self.vljepa_cfg = None
        self.vljepa_caption_query = _cfg(
            opt, "vljepa_caption_query", "Describe the image in detail."
        )
        self.vljepa_num_frames = _cfg_int(opt, "vljepa_num_frames", 1)
        self.vljepa_image_size = _cfg_int(opt, "vljepa_image_size", 256)
        self.vljepa_dtype = _cfg(opt, "vljepa_dtype", torch.bfloat16)
        self.vljepa_query_ids = None
        self.vljepa_query_mask = None

        if self.use_vljepa_condition:
            vljepa_ckpt = _cfg(opt, "vljepa_checkpoint_path", None)
            if not vljepa_ckpt or not os.path.exists(vljepa_ckpt):
                raise FileNotFoundError(
                    "use_vljepa_condition=true requires an existing vljepa_checkpoint_path."
                )
            logger.info(f"Loading Open-VLJEPA from {vljepa_ckpt}")
            self.vljepa_model, self.vljepa_cfg = load_open_vljepa(
                vljepa_ckpt,
                "cpu",
                dtype=self.vljepa_dtype,
            )
            self.vljepa_query_tokenizer, _ = load_vljepa_tokenizers(self.vljepa_cfg)
            self.vjepa = None
            self.dino = None
            self.clip = None
        elif self.use_vjepa_condition:
            logger.info(
                f"Loading V-JEPA 2.1 encoder from torch.hub entry {self.vjepa_hub_entry}"
            )
            self.vjepa = load_vjepa2_model(
                self.vjepa_hub_entry,
                "cpu",
                self.vjepa_dtype,
                offload=False,
                pretrained=self.vjepa_pretrained,
            )
            if self.vjepa_checkpoint_path and os.path.exists(self.vjepa_checkpoint_path):
                logger.info(f"Loading V-JEPA checkpoint from {self.vjepa_checkpoint_path}")
                load_vjepa2_weights(self.vjepa, self.vjepa_checkpoint_path)
            self.vjepa.requires_grad_(False)
            self.vjepa.eval()
            self.dino = None
            self.clip = None
            self.vljepa_model = None
        elif self.use_dino_condition:
            logger.info(f"Loading DINOv2 encoder from {self.dino_model_name}")
            self.dino = load_dinov2_model(
                self.dino_model_name,
                "cpu",
                self.dino_dtype,
            )
            self.vjepa = None
            self.clip = None
            self.vljepa_model = None
        elif self.use_clip_condition:
            logger.info(f"Loading CLIP vision encoder from {self.clip_model_name}")
            self.clip = load_clip_model(
                self.clip_model_name,
                "cpu",
                self.clip_dtype,
            )
            self.vjepa = None
            self.dino = None
            self.vljepa_model = None

        swinir_path = _cfg(opt, "swinir_model_path", None)
        self.swinir = None
        if swinir_path and os.path.exists(swinir_path):
            logger.info(f"Loading SwinIR from {swinir_path}")
            self.swinir = load_swinir("cpu", swinir_path, offload=False)

        revision = _cfg(opt, "revision", None)
        variant = _cfg(opt, "variant", None)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)

        if self.vjepa is not None:
            self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
        if self.dino is not None:
            self.dino = self.dino.to(self.accelerator.device, dtype=self.dino_dtype)
        if self.clip is not None:
            self.clip = self.clip.to(self.accelerator.device, dtype=self.clip_dtype)
        if self.vljepa_model is not None:
            self.vljepa_model = self.vljepa_model.to(
                self.accelerator.device, dtype=self.vljepa_dtype
            )
            max_query_len = int(self.vljepa_cfg.get("data", {}).get("max_query_len", 512))
            q_ids, q_mask = tokenize_vljepa_query(
                self.vljepa_caption_query,
                self.vljepa_query_tokenizer,
                max_query_len,
                self.accelerator.device,
            )
            self.vljepa_query_ids = q_ids
            self.vljepa_query_mask = q_mask
        if self.swinir is not None:
            self.swinir = self.swinir.to(self.accelerator.device)

        (
            self.num_latent_slots,
            self.latent_token_dim,
            self.latent_spatial_divisor,
            self.lat_h,
            self.lat_w,
        ) = get_latent_token_layout(
            self.train_crop_size,
            self.vae,
            transformer_config=transformer_config,
        )
        logger.info(
            "VAE token layout: num_slots=%d token_dim=%d crop=%d divisor=%d grid=%dx%d",
            self.num_latent_slots,
            self.latent_token_dim,
            self.train_crop_size,
            self.latent_spatial_divisor,
            self.lat_h,
            self.lat_w,
        )

        projector_hidden = _cfg_int(opt, "vljepa_projector_hidden", 2048)
        if self.vljepa_model is not None:
            embed_dim = VLJEPA_EMBED_DIM
        elif self.vjepa is not None:
            embed_dim = self.vjepa_feature_dim
        elif self.dino is not None:
            embed_dim = self.dino_feature_dim
        else:
            embed_dim = self.clip_feature_dim

        self.projector = PatchCrossAttnProjector(
            embed_dim=embed_dim,
            seq_dim=self.latent_token_dim,
            num_slots=self.num_latent_slots,
            hidden_dim=projector_hidden,
        )
        self.projector = self.projector.to(self.accelerator.device, dtype=weight_dtype)
        self.models.append(self.projector)

        if self.is_train:
            self.criterion = Loss(opt.train.loss).to(self.accelerator.device)

        self._logged_random_baseline = False

        logger.info(
            "ContextProjectorAlign: source_image=%s decode_tokens=%s extractor=(vljepa=%s vjepa=%s dino=%s clip=%s)",
            self.source_image,
            self.decode_tokens,
            self.use_vljepa_condition,
            self.use_vjepa_condition,
            self.use_dino_condition,
            self.use_clip_condition,
        )
        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"{m.__class__.__name__}: trainable={trainable:,} / total={all_param:,} "
                f"({100 * trainable / all_param:.1f}%)"
            )

    def setup_optimizers(self):
        opt = self.opt.train.optim
        optimizer_type = _cfg(opt, "type", "adamw").lower()
        trainable_params = [p for p in self.projector.parameters() if p.requires_grad]

        if optimizer_type == "adafactor":
            from transformers import Adafactor

            optimizer = Adafactor(
                trainable_params,
                lr=opt.learning_rate,
                relative_step=_cfg_bool(opt, "relative_step", False),
                scale_parameter=_cfg_bool(opt, "scale_parameter", False),
                warmup_init=_cfg_bool(opt, "warmup_init", False),
            )
        else:
            use_8bit = _cfg_bool(opt, "use_8bit_adam", False)
            if use_8bit:
                try:
                    import bitsandbytes as bnb

                    optimizer_class = bnb.optim.AdamW8bit
                except ImportError:
                    optimizer_class = torch.optim.AdamW
            else:
                optimizer_class = torch.optim.AdamW

            optimizer = optimizer_class(
                trainable_params,
                lr=opt.learning_rate,
                betas=(opt.adam_beta1, opt.adam_beta2),
                weight_decay=_cfg(opt, "adam_weight_decay", 0.01),
                eps=_cfg(opt, "adam_epsilon", 1e-8),
            )
        self.optimizers.append(optimizer)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    def _run_swinir(self, lq_01: torch.Tensor) -> torch.Tensor:
        if self.swinir is None:
            return lq_01
        _, _, h, w = lq_01.shape
        window_size = 8
        unshuffle_scale = 1
        required_multiple = max(1, window_size * unshuffle_scale)
        pad_h = (required_multiple - (h % required_multiple)) % required_multiple
        pad_w = (required_multiple - (w % required_multiple)) % required_multiple
        if pad_h or pad_w:
            lq_01 = F.pad(lq_01, (0, pad_w, 0, pad_h), mode="reflect")
        pre_01 = self.swinir(lq_01).detach().clamp(0.0, 1.0)
        return pre_01[:, :, :h, :w]

    def _resolve_source_image(
        self,
        gt_image: torch.Tensor,
        lq_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (source_rgb in [-1,1], source_01 in [0,1])."""
        if self.source_image == "gt":
            source_rgb = gt_image
        elif self.source_image == "lq":
            source_rgb = lq_image
        elif self.source_image == "pre":
            if self.swinir is None:
                raise ValueError("source_image=pre requires swinir_model_path.")
            lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            pre_01 = self._run_swinir(lq_01)
            source_rgb = pre_01 * 2.0 - 1.0
        else:
            raise ValueError(f"Unknown source_image={self.source_image!r}")
        source_01 = torch.clamp((source_rgb.float() + 1.0) / 2.0, 0.0, 1.0)
        return source_rgb, source_01

    def _extract_source_tokens(self, source_01: torch.Tensor) -> torch.Tensor:
        if self.vljepa_model is not None:
            return self._extract_vljepa_embedding(source_01)
        if self.vjepa is not None:
            return self._extract_vision_features(source_01)
        if self.dino is not None:
            return self._extract_dino_features(source_01)
        return self._extract_clip_features(source_01)

    def _extract_vision_features(self, pre_01: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            vjepa_pixels = vjepa_from_unit_tensor(
                pre_01,
                size=self.vjepa_image_size,
                device=self.accelerator.device,
                out_dtype=self.vjepa_dtype,
            )
            return extract_vjepa_tokens(
                self.vjepa,
                vjepa_pixels,
                device=self.accelerator.device,
                dtype=self.vjepa_dtype,
            )

    def _extract_dino_features(self, pre_01: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            dino_pixels = vjepa_from_unit_tensor(
                pre_01,
                size=self.dino_image_size,
                device=self.accelerator.device,
                out_dtype=self.dino_dtype,
            )
            return extract_dino_tokens(
                self.dino,
                dino_pixels,
                device=self.accelerator.device,
                dtype=self.dino_dtype,
            )

    def _extract_clip_features(self, pre_01: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            clip_pixels = clip_from_unit_tensor(
                pre_01,
                size=self.clip_image_size,
                device=self.accelerator.device,
                out_dtype=self.clip_dtype,
            )
            return extract_clip_tokens(
                self.clip,
                clip_pixels,
                device=self.accelerator.device,
                dtype=self.clip_dtype,
            )

    def _extract_vljepa_embedding(self, pre_01: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            pixel_values = preprocess_vljepa_image(
                pre_01,
                num_frames=self.vljepa_num_frames,
                image_size=self.vljepa_image_size,
                device=self.accelerator.device,
                out_dtype=self.vljepa_dtype,
            )
            query_ids = self.vljepa_query_ids
            query_mask = self.vljepa_query_mask
            if query_ids.shape[0] == 1 and pre_01.shape[0] > 1:
                query_ids = query_ids.expand(pre_01.shape[0], -1)
                query_mask = query_mask.expand(pre_01.shape[0], -1)
            return predict_semantic_embedding(
                self.vljepa_model,
                pixel_values,
                query_ids,
                query_mask,
            )

    def _compute_target_tokens(self, gt_image: torch.Tensor):
        with torch.inference_mode():
            target_tokens, patchified_latents = encode_vae_packed_tokens(
                self.vae,
                gt_image.to(self.weight_dtype),
                flux2_pipe_cls=self.flux2_pipe_cls,
                generator=None,
            )
        # Clone so MSE loss can use targets outside inference_mode autograd.
        return target_tokens.clone(), patchified_latents.clone()

    def _token_cosine_mean(
        self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        pred = F.normalize(pred_tokens.float(), dim=-1)
        target = F.normalize(target_tokens.float(), dim=-1)
        return (pred * target).sum(dim=-1).mean()

    def _forward_align(
        self,
        gt_image: torch.Tensor,
        lq_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, source_01 = self._resolve_source_image(gt_image, lq_image)
        target_tokens, patchified_latents = self._compute_target_tokens(gt_image)

        with torch.inference_mode():
            source_tokens = self._extract_source_tokens(source_01)

        pred_tokens = self.projector(source_tokens.to(self.weight_dtype).clone())
        return pred_tokens, target_tokens, patchified_latents, source_01

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        gt_image = self.sample[gt_key].clip(-1, 1)
        lq_image = self.sample[lq_key].clip(-1, 1)

        if gt_image.shape[1] == 1:
            gt_image = einops.repeat(gt_image, "b 1 h w -> b 3 h w")
        if lq_image.shape[1] == 1:
            lq_image = einops.repeat(lq_image, "b 1 h w -> b 3 h w")

        pred_tokens, target_tokens, _, _ = self._forward_align(gt_image, lq_image)
        losses = self.criterion(pred_tokens, target_tokens)

        if (
            self.global_step == 0
            and not self._logged_random_baseline
            and self.accelerator.is_main_process
        ):
            with torch.inference_mode():
                zero_pred = torch.zeros_like(target_tokens)
                random_baseline = F.mse_loss(zero_pred.float(), target_tokens.float())
            log_metric(
                self.accelerator,
                {"token_mse_random": random_baseline},
                self.global_step,
            )
            self._logged_random_baseline = True

        self.accelerator.backward(losses["all"])
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.projector.parameters(),
                getattr(self.opt.train.optim, "max_grad_norm", 1.0),
            )
            self.optimizers[0].step()
            self.optimizers[0].zero_grad()

        token_cosine = self._token_cosine_mean(pred_tokens, target_tokens)
        if self.accelerator.is_main_process:
            log_metric(
                self.accelerator,
                {
                    "token_mse": losses["all"].detach(),
                    "token_cosine": token_cosine.detach(),
                },
                self.global_step,
            )

        return losses

    @torch.no_grad()
    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)

        gt = self.sample[gt_key].clip(-1, 1)
        lq = self.sample[lq_key].clip(-1, 1)
        bsz = gt.shape[0]

        if gt.shape[1] == 1:
            gt = einops.repeat(gt, "b 1 h w -> b 3 h w")
        if lq.shape[1] == 1:
            lq = einops.repeat(lq, "b 1 h w -> b 3 h w")

        pred_tokens, target_tokens, patchified_latents, source_01 = self._forward_align(
            gt, lq
        )
        token_mse = F.mse_loss(pred_tokens.float(), target_tokens.float())
        token_cosine = self._token_cosine_mean(pred_tokens, target_tokens)
        if self.accelerator.is_main_process:
            log_metric(
                self.accelerator,
                {
                    "val_token_mse": token_mse,
                    "val_token_cosine": token_cosine,
                },
                self.global_step,
            )

        if not self.decode_tokens:
            return idx + bsz

        teacher_decode = decode_packed_vae_tokens(
            target_tokens,
            patchified_latents,
            self.vae,
            flux2_pipe_cls=self.flux2_pipe_cls,
        )
        pred_decode = decode_packed_vae_tokens(
            pred_tokens,
            patchified_latents,
            self.vae,
            flux2_pipe_cls=self.flux2_pipe_cls,
        )

        val_metrics = getattr(self.opt.val, "metrics", None)
        if val_metrics and self.accelerator.is_main_process:
            gt_np = (gt.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            pred_np = (pred_decode.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            teacher_np = (teacher_decode.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            log_metrics(gt_np, pred_np, val_metrics, self.accelerator, self.global_step)
            log_metrics(
                gt_np,
                teacher_np,
                val_metrics,
                self.accelerator,
                self.global_step,
                comment="teacher_",
            )

        if getattr(self.opt.val, "save_images", False) and self.accelerator.is_main_process:
            source_np = source_01.cpu().float().numpy().clip(0, 1)
            teacher_np = (teacher_decode.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            pred_np = (pred_decode.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
            for i in range(bsz):
                idx += 1
                panels = [source_np[i], teacher_np[i], pred_np[i]]
                for j in range(len(panels)):
                    if panels[j].shape[0] == 1:
                        panels[j] = einops.repeat(panels[j], "1 h w -> 3 h w")
                images = np.stack(panels)
                log_image(
                    self.opt,
                    self.accelerator,
                    images,
                    f"{idx:04d}_src_teacher_pred",
                    self.global_step,
                )
        else:
            idx += bsz

        return idx

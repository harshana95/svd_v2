import copy
import gc
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2
from tqdm import tqdm

from models.base_model import BaseModel
from utils import log_image, log_metrics

from .helpers import (
    PROMPT,
    calculate_shift,
    decode_latents_flux,
    encode_vae_image,
    get_flux2_pipeline_class,
    get_noisy_model_input_and_timesteps,
    initialize_transformer_lora,
    prepare_image_latents,
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
    VLJEPACaptionReadout,
    infer_num_text_slots,
    load_open_vljepa,
    load_ram_tag_candidates,
    load_vljepa_tokenizers,
    predict_semantic_embedding,
    preprocess_vljepa_image,
    tokenize_vljepa_query,
)
from .vljepa_projector import VLJEPAProjector
from .src.flux.lucidflux import (
    load_precomputed_embeddings,
    load_swinir,
)


class ContextFluxBaseline_model(BaseModel):
    def __init__(self, opt, logger):
        super(ContextFluxBaseline_model, self).__init__(opt, logger)

        self.use_condition_image_for_transformer = getattr(opt, "use_condition_image_for_transformer", True)
        self.use_vljepa_condition = getattr(opt, "use_vljepa_condition", False)
        self.use_vjepa_condition = getattr(opt, "use_vjepa_condition", True)
        if self.use_vljepa_condition and self.use_vjepa_condition:
            logger.warning(
                "Both use_vljepa_condition and use_vjepa_condition are enabled; "
                "preferring VL-JEPA semantic conditioning and disabling V-JEPA patch tokens."
            )
            self.use_vjepa_condition = False

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        # V-JEPA hub weights use fp32 attention; bf16 inputs cause Q/K/V dtype mismatches.
        self.vjepa_dtype = getattr(opt, "vjepa_dtype", torch.float32)

        pretrained = opt.pretrained_model_name_or_path
        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)
        transformer_config = Flux2Transformer2DModel.load_config(
            pretrained, subfolder="transformer"
        )
        # DictAsMember returns None for missing keys via __getattr__, so use .get().
        text_in_dim = opt.get("text_in_dim")
        if text_in_dim is None:
            text_in_dim = transformer_config.get("joint_attention_dim")
        if text_in_dim is None:
            text_in_dim = 15360
        default_guidance_scale = 1.0 if "klein" in pretrained.lower() else 4.0

        self.vjepa_hub_entry = getattr(opt, "vjepa_hub_entry", "vjepa2_1_vit_large_384")
        self.vjepa_image_size = getattr(opt, "vjepa_image_size", 384)
        self.vjepa_pretrained = getattr(opt, "vjepa_pretrained", True) 
        self.vjepa_checkpoint_path = getattr(opt, "vjepa_checkpoint_path", None)

        self.vjepa_feature_dim = get_vjepa_feature_dim(
            self.vjepa_hub_entry,
            fallback_dim=getattr(opt, "vjepa_feature_dim", 1024),
        )

        self.vljepa_model = None
        self.vljepa_readout = None
        self.vljepa_query_tokenizer = None
        self.vljepa_cfg = None
        self.vljepa_caption_query = getattr(
            opt, "vljepa_caption_query", "Describe the image in detail."
        )
        self.vljepa_num_frames = int(getattr(opt, "vljepa_num_frames", 1))
        self.vljepa_image_size = int(getattr(opt, "vljepa_image_size", 256))
        self.vljepa_dtype = getattr(opt, "vljepa_dtype", torch.bfloat16)
        self.vljepa_query_ids = None
        self.vljepa_query_mask = None

        if self.use_vljepa_condition:
            vljepa_ckpt = getattr(opt, "vljepa_checkpoint_path", None)
            if not vljepa_ckpt or not os.path.exists(vljepa_ckpt):
                raise FileNotFoundError(
                    "use_vljepa_condition=true requires an existing vljepa_checkpoint_path "
                    "(download cun-bjy/open-vljepa best.pt)."
                )
            logger.info(f"Loading Open-VLJEPA from {vljepa_ckpt}")
            self.vljepa_model, self.vljepa_cfg = load_open_vljepa(
                vljepa_ckpt,
                "cpu",
                dtype=self.vljepa_dtype,
            )
            self.vljepa_query_tokenizer, y_tokenizer = load_vljepa_tokenizers(self.vljepa_cfg)
            readout_vocab = getattr(opt, "vljepa_readout_vocab", "ram_tags")
            enable_readout = bool(getattr(opt, "vljepa_enable_readout", True))
            if enable_readout and readout_vocab not in (None, "", "none", "None"):
                if readout_vocab == "ram_tags":
                    candidates = load_ram_tag_candidates()
                else:
                    with open(readout_vocab, encoding="utf-8") as f:
                        candidates = [line.strip() for line in f if line.strip()]
                max_caption_len = int(
                    self.vljepa_cfg.get("data", {}).get("max_caption_len", 128)
                )
                try:
                    self.vljepa_readout = VLJEPACaptionReadout(
                        self.vljepa_model,
                        y_tokenizer,
                        candidates,
                        device="cpu",
                        max_caption_len=max_caption_len,
                    )
                    logger.info(
                        "VL-JEPA caption readout enabled with %d frozen candidate tags",
                        len(candidates),
                    )
                except OSError as exc:
                    logger.warning(
                        "VL-JEPA caption readout disabled (Y-encoder access failed): %s",
                        exc,
                    )
                    self.vljepa_readout = None
            else:
                logger.info("VL-JEPA caption readout disabled via config.")
            self.vjepa = None
            logger.info("V-JEPA patch conditioning disabled (VL-JEPA semantic path active).")
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
        else:
            logger.info("V-JEPA conditioning disabled via `use_vjepa_condition`.")
            self.vjepa = None
        
        swinir_path = getattr(opt, "swinir_model_path", None)
        self.swinir = None
        if swinir_path and os.path.exists(swinir_path):
            logger.info(f"Loading SwinIR from {swinir_path}")
            self.swinir = load_swinir("cpu", swinir_path, offload=False)

        # loading precomputed embeddings
        embeddings_path = getattr(opt, "precomputed_embeddings_path", None)
        if embeddings_path and os.path.exists(embeddings_path):
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_vec = embeds.get("vec")
            self.register_buffer_txt_ids = embeds.get("txt_ids")
        else:
            logger.warning(
                "No precomputed embeddings path found. Will generate null embeddings at first forward pass."
            )
            self.register_buffer_txt = None
            self.register_buffer_vec = None
            self.register_buffer_txt_ids = None
        
        # loading Flux 2 transformer and vae
        self.guidance_scale = getattr(opt, "guidance_scale", default_guidance_scale)
        self.discrete_flow_shift = getattr(opt, "discrete_flow_shift", 3.1582)
        revision = getattr(opt, "revision", None)
        variant = getattr(opt, "variant", None)
        base_transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant
        )
        base_transformer.requires_grad_(False)
        base_transformer.eval()
        if getattr(opt, "transformer_gradient_checkpointing", False):
            if hasattr(base_transformer, "enable_gradient_checkpointing"):
                base_transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled for frozen transformer")
        else:
            logger.info(
                "Transformer gradient checkpointing disabled (faster training, higher VRAM)"
            )
        self.lora_finetune = bool(getattr(opt.train, "lora_finetune", False))
        if self.lora_finetune:
            lora_rank = int(getattr(opt.train, "lora_rank", 8))
            lora_alpha = int(getattr(opt.train, "lora_alpha", max(1, lora_rank * 2)))
            lora_dropout = float(getattr(opt.train, "lora_dropout", 0.0))
            lora_target_modules = getattr(opt.train, "lora_target_modules", None)
            self.transformer = initialize_transformer_lora(
                base_transformer,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_patterns=lora_target_modules,
            )
            for name, param in self.transformer.named_parameters():
                param.requires_grad = "lora" in name
            logger.info(
                "Enabled Flux2 LoRA adapters in ContextFluxBaseline_model: "
                f"rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout} target_modules={lora_target_modules}"
            )
        else:
            self.transformer = base_transformer
            logger.info(
                "Transformer LoRA disabled in ContextFluxBaseline_model (set train.lora_finetune=true to enable)."
            )

        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.transformer_context_dim = getattr(
            self.transformer.config, "joint_attention_dim", text_in_dim
        )
        self.use_transformer_guidance = getattr(
            self.transformer.config, "guidance_embeds", True
        )
        self.transformer_rope_dims = len(
            getattr(self.transformer.config, "axes_dims_rope", (32, 32, 32, 32))
        )

        self.transformer = self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)

        if self.vjepa is not None:
            self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
        if self.vljepa_model is not None:
            self.vljepa_model = self.vljepa_model.to(
                self.accelerator.device, dtype=self.vljepa_dtype
            )
            if self.vljepa_readout is not None:
                self.vljepa_readout.device = self.accelerator.device
                self.vljepa_readout.model = self.vljepa_readout.model.to(
                    self.accelerator.device, dtype=self.vljepa_dtype
                )
                self.vljepa_readout.candidate_embeds = (
                    self.vljepa_readout.candidate_embeds.to(self.accelerator.device)
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


        # Initializing trainable modules

        num_text_slots = infer_num_text_slots(
            getattr(opt, "precomputed_embeddings_path", None),
            default=int(getattr(opt, "vljepa_num_text_slots", 512) or 512),
        )
        if getattr(opt, "vljepa_num_text_slots", None) not in (None, "~"):
            num_text_slots = int(opt.vljepa_num_text_slots)
        self.vljepa_num_text_slots = num_text_slots

        if self.vljepa_model is not None:
            projector_hidden = int(getattr(opt, "vljepa_projector_hidden", 2048))
            self.vljepa_projector = VLJEPAProjector(
                vljepa_dim=VLJEPA_EMBED_DIM,
                seq_dim=self.transformer_context_dim,
                num_slots=num_text_slots,
                hidden_dim=projector_hidden,
            )
            self.vljepa_projector.requires_grad_(True)
            self.vljepa_projector = self.vljepa_projector.to(
                self.accelerator.device, dtype=weight_dtype
            )
            self.models.append(self.vljepa_projector)
        elif self.vjepa is not None:
            self.jepa_projector = torch.nn.Linear(self.vjepa_feature_dim, self.transformer_context_dim)
            self.jepa_projector.requires_grad_(True)
            self.jepa_projector = self.jepa_projector.to(self.accelerator.device, dtype=weight_dtype)
            self.models.append(self.jepa_projector)
        if self.lora_finetune:
            self.models.append(self.transformer)

        logger.info(
            "Condition toggles: use_vljepa_condition=%s, use_vjepa_condition=%s, "
            "use_condition_image_for_transformer=%s",
            self.use_vljepa_condition,
            self.use_vjepa_condition,
            self.use_condition_image_for_transformer,
        )

        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"{m.__class__.__name__}: trainable={trainable:,} / total={all_param:,} "
                f"({100 * trainable / all_param:.1f}%)"
            )

    def _run_swinir(self, lq_01: torch.Tensor) -> torch.Tensor:
        if self.swinir is None:
            return lq_01

        _, _, h, w = lq_01.shape
        window_size = int(getattr(self.swinir, "window_size", 1))
        unshuffle_scale = int(getattr(self.swinir, "unshuffle_scale", 1) or 1)
        required_multiple = max(1, window_size * unshuffle_scale)

        pad_h = (required_multiple - (h % required_multiple)) % required_multiple
        pad_w = (required_multiple - (w % required_multiple)) % required_multiple
        if pad_h or pad_w:
            lq_01 = F.pad(lq_01, (0, pad_w, 0, pad_h), mode="reflect")

        pre_01 = self.swinir(lq_01).detach().clamp(0.0, 1.0)
        return pre_01[:, :, :h, :w]

    def _ensure_null_embeddings(self):
        if self.register_buffer_txt is not None and self.register_buffer_txt_ids is not None:
            return
        self.logger.info("Generating null-text embeddings via Flux 2 text encoders...")
        pipe = self.flux2_pipe_cls.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=self.weight_dtype,
        ).to(self.accelerator.device)
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                [PROMPT], num_images_per_prompt=1
            )

        self.register_buffer_txt = prompt_embeds.cpu()
        self.register_buffer_txt_ids = text_ids.cpu()
        self.register_buffer_vec = None
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def setup_optimizers(self):
        opt = self.opt.train.optim
        optimizer_type = getattr(opt, "type", "adamw").lower()
        trainable_params = []
        if self.vljepa_model is not None:
            trainable_params.extend(
                p for p in self.vljepa_projector.parameters() if p.requires_grad
            )
        elif self.vjepa is not None:
            trainable_params.extend(
                p for p in self.jepa_projector.parameters() if p.requires_grad
            )
        if self.lora_finetune:
            trainable_params.extend(
                p for p in self.transformer.parameters() if p.requires_grad
            )

        if optimizer_type == "adafactor":
            from transformers import Adafactor

            optimizer = Adafactor(
                trainable_params,
                lr=opt.learning_rate,
                relative_step=getattr(opt, "relative_step", False),
                scale_parameter=getattr(opt, "scale_parameter", False),
                warmup_init=getattr(opt, "warmup_init", False),
            )
        else:
            use_8bit = getattr(opt, "use_8bit_adam", False)
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
                weight_decay=getattr(opt, "adam_weight_decay", 0.01),
                eps=getattr(opt, "adam_epsilon", 1e-8),
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

    def _extract_vision_features(self, pre_01: torch.Tensor) -> torch.Tensor:
        if self.vjepa is None:
            raise RuntimeError("V-JEPA is disabled but `_extract_vision_features` was called.")
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

    def _extract_vljepa_embedding(self, pre_01: torch.Tensor) -> torch.Tensor:
        if self.vljepa_model is None:
            raise RuntimeError("VL-JEPA is disabled but `_extract_vljepa_embedding` was called.")
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

    def _encode_vljepa_conditioning(
        self,
        semantic_embeds: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        txt = self.vljepa_projector(semantic_embeds.to(self.weight_dtype))
        txt_ids = self.flux2_pipe_cls._prepare_text_ids(txt)
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0)
        if txt_ids.shape[0] == 1 and batch_size > 1:
            txt_ids = txt_ids.expand(batch_size, -1, -1)
        return txt.to(device), txt_ids.to(device=device, dtype=self.weight_dtype)

    def _log_vljepa_readout(self, semantic_embeds: torch.Tensor):
        if self.vljepa_readout is None or not self.accelerator.is_main_process:
            return
        captions = self.vljepa_readout.retrieve(semantic_embeds)
        preview = captions[0] if captions else ""
        self.logger.info("VL-JEPA readout (sample 0): %s", preview)

    def _prepare_text_inputs(self, batch_size: int, device: torch.device):
        self._ensure_null_embeddings()
        txt = self.register_buffer_txt.to(device).expand(batch_size, -1, -1).to(self.weight_dtype)
        txt_ids = self.register_buffer_txt_ids.to(device).to(self.weight_dtype)
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
        elif txt_ids.shape[0] == 1 and batch_size > 1:
            txt_ids = txt_ids.expand(batch_size, -1, -1)
        return txt, txt_ids

    def _transformer_guidance(self, batch_size: int, device: torch.device):
        if not self.use_transformer_guidance:
            return None
        return torch.full(
            (batch_size,),
            self.guidance_scale,
            device=device,
            dtype=self.weight_dtype,
        )

    def optimize_parameters(self):
        if not self.use_vljepa_condition:
            self._ensure_null_embeddings()

        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        gt_image = self.sample[gt_key].clip(-1, 1)
        lq_image = self.sample[lq_key].clip(-1, 1)
        bsz = gt_image.shape[0]

        if gt_image.shape[1] == 1:
            gt_image = einops.repeat(gt_image, "b 1 h w -> b 3 h w")
        if lq_image.shape[1] == 1:
            lq_image = einops.repeat(lq_image, "b 1 h w -> b 3 h w")

        semantic_embeds = None
        vision_image_pre_fts = None

        # Frozen encoders: single batched VAE encode + SwinIR/V-JEPA/VL-JEPA under inference_mode.
        with torch.inference_mode():
            pixel_latents = encode_vae_image(
                self.vae, gt_image.to(self.weight_dtype), None, flux2_pipe_cls=self.flux2_pipe_cls
            )
            if self.use_condition_image_for_transformer:
                image_latents, image_latent_ids = prepare_image_latents(
                    vae=self.vae,
                    ref_images=lq_image.to(self.weight_dtype),
                    batch_size=bsz,
                    generator=None,
                    device=self.accelerator.device,
                    dtype=self.weight_dtype,
                    flux2_pipe_cls=self.flux2_pipe_cls,
                )
            else:
                image_latents = None
                image_latent_ids = None

            lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            if self.swinir is not None:
                pre_01 = self._run_swinir(lq_01)
            else:
                pre_01 = lq_01

            if self.use_vljepa_condition:
                semantic_embeds = self._extract_vljepa_embedding(pre_01)
                if self.global_step == 0:
                    self._log_vljepa_readout(semantic_embeds)
            elif self.use_vjepa_condition:
                vision_image_pre_fts = self._extract_vision_features(pre_01)

        noise = torch.randn_like(pixel_latents)
        noisy_latents, timesteps, _ = get_noisy_model_input_and_timesteps(
            self.noise_scheduler_copy,
            pixel_latents,
            noise,
            pixel_latents.device,
            discrete_flow_shift=self.discrete_flow_shift,
        )

        transformer_latents = noisy_latents
        packed_transformer_input = self.flux2_pipe_cls._pack_latents(transformer_latents)
        transformer_image_ids = self.flux2_pipe_cls._prepare_latent_ids(
            transformer_latents
        ).to(device=self.accelerator.device, dtype=self.weight_dtype)

        latent_token_count = packed_transformer_input.shape[1]

        # prepare text / semantic conditioning
        if self.use_vljepa_condition:
            encoder_hidden_states, encoder_txt_ids = self._encode_vljepa_conditioning(
                semantic_embeds, bsz, self.accelerator.device
            )
        else:
            txt, txt_ids = self._prepare_text_inputs(bsz, self.accelerator.device)
            if self.use_vjepa_condition:
                jepa_embeds = self.jepa_projector(vision_image_pre_fts.to(self.weight_dtype))
                jepa_ids = self.flux2_pipe_cls._prepare_text_ids(jepa_embeds)
                if jepa_ids.ndim == 2:
                    jepa_ids = jepa_ids.unsqueeze(0)
                if jepa_ids.shape[0] == 1 and txt_ids.shape[0] > 1:
                    jepa_ids = jepa_ids.expand(txt_ids.shape[0], -1, -1)
                jepa_ids = jepa_ids.to(device=txt_ids.device, dtype=txt_ids.dtype)
                jepa_ids[:, :, 0] = 1 # set the first dimension to 1
                encoder_hidden_states = torch.cat([txt, jepa_embeds], dim=1)
                encoder_txt_ids = torch.cat([txt_ids, jepa_ids], dim=1)
            else:
                encoder_hidden_states = txt
                encoder_txt_ids = txt_ids

        transformer_guidance = self._transformer_guidance(bsz, self.accelerator.device)
        normalized_timesteps = (timesteps.float() / 1000).to(self.weight_dtype)

        latent_model_input = packed_transformer_input
        latent_image_ids = transformer_image_ids
        if image_latents is not None:
            latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
            latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

        model_pred = self.transformer(
            hidden_states=latent_model_input.to(self.weight_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weight_dtype),
            timestep=normalized_timesteps,
            guidance=transformer_guidance,
            txt_ids=encoder_txt_ids.to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]
        model_pred = model_pred[:, :latent_token_count]
        model_pred = self.flux2_pipe_cls._unpack_latents_with_ids(
            model_pred,
            transformer_image_ids,
        )

        target = noise - pixel_latents
        loss = F.mse_loss(model_pred.float(), target.float())
        # if not self.use_vjepa_condition:
        #     # Keep backward() valid when V-JEPA branch is disabled and the projector is unused.
        #     loss = loss + (0.0 * self.jepa_projector.weight.sum())

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            clip_params = []
            if self.vljepa_model is not None:
                clip_params.extend(list(self.vljepa_projector.parameters()))
            elif self.vjepa is not None:
                clip_params.extend(list(self.jepa_projector.parameters()))
            if self.lora_finetune:
                clip_params.extend(
                    p for p in self.transformer.parameters() if p.requires_grad
                )
            self.accelerator.clip_grad_norm_(
                clip_params,
                getattr(self.opt.train.optim, "max_grad_norm", 1.0),
            )
            self.optimizers[0].step()
            self.optimizers[0].zero_grad()

        return {"all": loss}

    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()

        for model in self.models:
            model.eval()

        idx = 0
        for batch in tqdm(self.test_dataloader, desc="Validation"):
            idx = self.validate_step(
                batch,
                idx,
                self.test_dataloader.dataset.lq_key,
                self.test_dataloader.dataset.gt_key,
            )
            if idx >= self.max_val_steps:
                break

        gc.collect()
        torch.cuda.empty_cache()
        for model in self.models:
            model.train()

    @torch.no_grad()
    def forwardpass(
        self, lq_image: torch.Tensor, height: int, width: int, num_steps: int = 50
    ):
        if not self.use_vljepa_condition:
            self._ensure_null_embeddings()
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = lq_image.shape[0]

        encoder_hidden_states = None
        encoder_txt_ids = None
        txt = None
        txt_ids = None
        semantic_embeds = None
        vision_image_pre_fts = None

        if not self.use_vljepa_condition:
            txt, txt_ids = self._prepare_text_inputs(bsz, device)

        # Match Flux2 prepare_latents() sizing without an extra VAE encode.
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_spatial_divisor = vae_scale_factor * 2
        lat_c = self.transformer.config.in_channels
        lat_h = int(height) // latent_spatial_divisor
        lat_w = int(width) // latent_spatial_divisor
        latents = torch.randn(bsz, lat_c, lat_h, lat_w, device=device, dtype=dtype)

        transformer_guidance = self._transformer_guidance(bsz, device)
        with torch.inference_mode():
            if self.use_condition_image_for_transformer:
                image_latents, image_latent_ids = prepare_image_latents(
                    vae=self.vae,
                    ref_images=lq_image.to(dtype),
                    batch_size=bsz,
                    generator=None,
                    device=device,
                    dtype=dtype,
                    flux2_pipe_cls=self.flux2_pipe_cls,
                )
            else:
                image_latents = None
                image_latent_ids = None

            lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            if self.swinir is not None:
                pre_01 = self._run_swinir(lq_01)
            else:
                pre_01 = lq_01

            if self.use_vljepa_condition:
                semantic_embeds = self._extract_vljepa_embedding(pre_01)
                encoder_hidden_states, encoder_txt_ids = self._encode_vljepa_conditioning(
                    semantic_embeds, bsz, device
                )
            elif self.use_vjepa_condition:
                vision_image_pre_fts = self._extract_vision_features(pre_01)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
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

        for t in tqdm(scheduler.timesteps):
            transformer_latents = latents
            packed_transformer_input = self.flux2_pipe_cls._pack_latents(transformer_latents)
            transformer_image_ids = self.flux2_pipe_cls._prepare_latent_ids(
                transformer_latents
            ).to(device=device, dtype=dtype)
            latent_token_count = packed_transformer_input.shape[1]

            timestep = t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0

            if self.use_vljepa_condition:
                step_encoder_hidden_states = encoder_hidden_states
                step_encoder_txt_ids = encoder_txt_ids
            elif self.use_vjepa_condition:
                jepa_embeds = self.jepa_projector(vision_image_pre_fts.to(self.weight_dtype))
                jepa_ids = self.flux2_pipe_cls._prepare_text_ids(jepa_embeds)
                if jepa_ids.ndim == 2:
                    jepa_ids = jepa_ids.unsqueeze(0)
                if jepa_ids.shape[0] == 1 and txt_ids.shape[0] > 1:
                    jepa_ids = jepa_ids.expand(txt_ids.shape[0], -1, -1)
                jepa_ids[:, :, 0] = 1 # set the first dimension to 1
                jepa_ids = jepa_ids.to(device=txt_ids.device, dtype=txt_ids.dtype)
                step_encoder_hidden_states = torch.cat([txt, jepa_embeds], dim=1)
                step_encoder_txt_ids = torch.cat([txt_ids, jepa_ids], dim=1)
            else:
                step_encoder_hidden_states = txt
                step_encoder_txt_ids = txt_ids

            latent_model_input = packed_transformer_input
            latent_image_ids = transformer_image_ids
            if image_latents is not None:
                latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
                latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=step_encoder_hidden_states.to(dtype),
                timestep=timestep,
                guidance=transformer_guidance,
                txt_ids=step_encoder_txt_ids.to(dtype),
                img_ids=latent_image_ids.to(dtype),
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latent_token_count]
            noise_pred = self.flux2_pipe_cls._unpack_latents_with_ids(
                noise_pred,
                transformer_image_ids,
            )
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = decode_latents_flux(latents, self.vae, flux2_pipe_cls=self.flux2_pipe_cls)
        return decoded.clamp(-1, 1)

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = getattr(self.opt.val, "num_inference_steps", 50)

        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        bsz = lq.shape[0]
        swinir_np = None

        if lq.shape[1] == 1:
            lq = einops.repeat(lq, "b 1 h w -> b 3 h w")

        out = self.forwardpass(
            lq, lq.shape[-2], lq.shape[-1], num_inference_steps
        )

        if self.swinir is not None:
            with torch.inference_mode():
                lq_01 = torch.clamp((lq.float() + 1.0) / 2.0, 0.0, 1.0)
                pre_01 = self._run_swinir(lq_01)
            swinir_np = pre_01.cpu().float().numpy().clip(0, 1)

        lq_np = (lq.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        gt_np = (gt.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        for i in range(bsz):
            idx += 1
            images = [lq_np[i], gt_np[i], out_np[i]]
            for j in range(len(images)):
                if images[j].shape[0] == 1:
                    images[j] = einops.repeat(images[j], "1 h w -> 3 h w")
            images = np.stack(images)
            images = np.clip(images, 0, 1)
            log_image(self.opt, self.accelerator, images, f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator, out_np[i:i+1], f"out_{idx:04d}", self.global_step)
            if swinir_np is not None:
                log_image(
                    self.opt,
                    self.accelerator,
                    swinir_np[i:i+1],
                    f"swinir_{idx:04d}",
                    self.global_step,
                )
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

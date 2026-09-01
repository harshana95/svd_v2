"""VAEBlurAlignFluxLoRA_model: Phase 2 of the VAEBlurAlign investigation.

Phase 1 (VAEBlurAlign_model) trained a LoRA-adapted FLUX.2 Klein VAE encoder
to produce latent tokens that recover a blurred sub-region given clean
surrounding context, judged by decoding those tokens through the VAE's own
FROZEN decoder. This phase asks a different question: instead of relying on
the VAE's own (fairly weak, ~0.001 MSE round-trip noise floor) decoder, can
FLUX.2's own generative power do better if it's given the SAME trained
encoder's tokens as conditioning and fine-tuned (rank-128 LoRA) to use them?

Pipeline, mirroring models/LucidFlux/LucidFlux2LoRAJEPA_model.py's
"raw image-latent conditioning" path (its `image_latents` branch, concatenated
into `hidden_states` -- NOT its separate vjepa `encoder_hidden_states` path,
which this model doesn't use at all):

  1. self.cond_vae: a SECOND, independent AutoencoderKLFlux2 instance, with
     the SAME LoRA structure (`lora_adapt_vae_encoder`, imported from
     VAEBlurAlign_model.py) re-applied and the phase-1 checkpoint's trained
     weights loaded into it. This encodes the "mixed" (context + blurred
     block) input into packed tokens -- the conditioning signal.
  2. Those tokens go into the FLUX2 transformer's `hidden_states`, concatenated
     alongside the noisy target latent's own tokens (via
     helpers.prepare_image_latents, which already returns tokens in the same
     packed-latent channel space `_pack_latents`/`_patchify_latents` produce
     for VAEBlurAlign's own encode path -- no projector needed, channel dims
     already match).
  3. FLUX2's transformer is LoRA-fine-tuned (rank from train.lora_rank, 128
     per this phase's config) with a standard rectified-flow / flow-matching
     MSE loss against the GT image's (separately, FROZEN plain-VAE-encoded)
     latent.

cond_vae's own trainability is config-gated via `vae_trainable` (default
False: frozen feature extractor, exactly reproducing phase 1's checkpoint).
Set it True to keep adapting the encoder jointly with the FLUX LoRA. Either
way, the VAE DECODER is never touched: `lora_adapt_vae_encoder` only ever
targets encoder-side modules, and cond_vae's decoder is never called here at
all (only self.vae's decoder is, for the final FLUX output and for the frozen
GT-target encoding -- see decode_latents_flux/encode_vae_image below).

Dataset: FullResPatchifierDataset, unchanged from phase 1 -- "mixed" (context
+ blurred block) is the conditioning input, "gt" is the flow-matching target,
matching the user's request to keep phase 2's input construction identical to
what cond_vae was actually trained on.
"""
import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2

from dataset import create_dataset
from models.base_model import BaseModel
from models.LucidFlux.helpers import (
    calculate_shift,
    decode_latents_flux,
    encode_vae_image,
    get_flux2_pipeline_class,
    get_noisy_model_input_and_timesteps,
    initialize_transformer_lora,
    prepare_image_latents,
)
from models.reFlux.VAEBlurAlign_model import lora_adapt_vae_encoder
from utils import log_image, log_metrics


def _cfg(opt, key, default=None):
    value = opt.get(key, default) if hasattr(opt, "get") else getattr(opt, key, default)
    return default if value in (None, "~") else value


class VAEBlurAlignFluxLoRA_model(BaseModel):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        pretrained = opt.pretrained_model_name_or_path
        revision = _cfg(opt, "revision", None)
        variant = _cfg(opt, "variant", None)
        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)

        # Null-text (unconditional) embeddings -- this model conditions on
        # cond_vae's image tokens, not on a caption, so the transformer's
        # encoder_hidden_states input is always the empty-prompt embedding,
        # generated lazily on first use and cached (same as
        # LucidFlux2LoRAJEPA_model, whose null-embedding handling this
        # reuses verbatim).
        self.register_buffer_txt = None
        self.register_buffer_txt_ids = None

        self.guidance_scale = float(_cfg(opt, "guidance_scale", 4.0))
        self.discrete_flow_shift = float(_cfg(opt, "discrete_flow_shift", 3.1582))

        # ---- FLUX2 transformer + LoRA (the thing actually being trained) ----
        base_transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant
        )
        base_transformer.requires_grad_(False)
        if _cfg(opt, "transformer_gradient_checkpointing", False):
            if hasattr(base_transformer, "enable_gradient_checkpointing"):
                base_transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled for LoRA transformer")

        lora_rank = int(_cfg(opt.train, "lora_rank", 128))
        if lora_rank > 0:
            lora_alpha = int(_cfg(opt.train, "lora_alpha", max(1, lora_rank * 2)))
            lora_dropout = float(_cfg(opt.train, "lora_dropout", 0.0))
            lora_target_modules = _cfg(opt.train, "lora_target_modules", None)

            self.transformer = initialize_transformer_lora(
                base_transformer,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_patterns=lora_target_modules,
            )
            for name, param in self.transformer.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            logger.info(f"Enabled Flux2 transformer LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
        else:
            # lora_rank<=0: the "original FLUX2, no training at all" baseline
            # -- transformer stays exactly base_transformer, already fully
            # frozen above. Only useful with is_train=false (run via
            # `trainer.py -opt ... -infer`, which skips the epoch loop
            # entirely and falls through to one final validation() pass --
            # see base_model.py's train(), the comment right above
            # `self.validation()` at the end). There is nothing to optimize,
            # so setup_optimizers() must never be called in this mode -- it
            # isn't, since BaseModel.prepare() only calls it when is_train.
            self.transformer = base_transformer
            logger.info(
                "lora_rank<=0: FLUX2 transformer is FULLY FROZEN (no LoRA) -- "
                "this is the 'original FLUX2, no fine-tuning' baseline. "
                "Run with is_train=false / -infer; there is nothing to train."
            )

        # ---- Frozen plain VAE: encodes GT targets, decodes final output ----
        # Never touched by any LoRA -- this is the exact frozen mapping the
        # rest of the pipeline (and inference) assumes, matching every other
        # FLUX2 model in this repo.
        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ---- Conditioning VAE: phase 1's trained encoder ----
        # A SEPARATE instance from self.vae -- this one gets the same LoRA
        # structure phase 1 used re-applied, then phase 1's trained weights
        # loaded into it. It is ONLY ever used to encode the conditioning
        # ("mixed") image; its decoder is never called anywhere in this file.
        self.vae_trainable = bool(_cfg(opt, "vae_trainable", False))
        vae_lora_rank = int(_cfg(opt, "vae_lora_rank", 128))
        vae_checkpoint_path = _cfg(opt, "vae_checkpoint_path", None)
        if not vae_checkpoint_path:
            raise ValueError(
                "vae_checkpoint_path is required -- point it at the phase-1 "
                "VAEBlurAlign checkpoint's VAE weights dir, e.g. "
                ".../VAEBlurAlign_.../checkpoint-6000/AutoencoderKLFlux2_1"
            )
        self.cond_vae = AutoencoderKLFlux2.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.cond_vae.requires_grad_(False)
        self.cond_vae = lora_adapt_vae_encoder(self.cond_vae, vae_lora_rank)
        state_dict_path = os.path.join(vae_checkpoint_path, "diffusion_pytorch_model.safetensors")
        state_dict = load_safetensors(state_dict_path)
        # strict=True: the LoRA structure re-applied above must exactly match
        # what phase 1 trained (same vae_lora_rank) or this will fail loudly
        # on a shape mismatch, rather than silently loading a wrong/partial
        # state.
        self.cond_vae.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded phase-1 VAEBlurAlign checkpoint into cond_vae from {state_dict_path}")
        if self.vae_trainable:
            for n, p in self.cond_vae.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
            self.models.append(self.cond_vae)
            logger.info("cond_vae LoRA params are TRAINABLE (vae_trainable=true) -- continuing phase-1 adaptation jointly with the FLUX LoRA")
        else:
            self.cond_vae.eval()
            logger.info("cond_vae is FROZEN (vae_trainable=false) -- used as a fixed feature extractor from the phase-1 checkpoint")

        self.noise_scheduler_copy = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained, subfolder="scheduler"
        )
        self.use_transformer_guidance = getattr(self.transformer.config, "guidance_embeds", True)

        self.transformer = self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.cond_vae = self.cond_vae.to(self.accelerator.device, dtype=weight_dtype)

        self.models.append(self.transformer)

        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"{m.__class__.__name__}: trainable={trainable:,} / total={all_param:,} "
                f"({100 * trainable / all_param:.3f}%)"
            )

    def setup_optimizers(self):
        opt = self.opt.train.optim
        trainable_params = [{"params": [p for p in self.transformer.parameters() if p.requires_grad],
                              "lr": opt.learning_rate}]
        if self.vae_trainable:
            vae_lr = float(_cfg(opt, "cond_vae_learning_rate", opt.learning_rate))
            trainable_params.append({"params": [p for p in self.cond_vae.parameters() if p.requires_grad],
                                      "lr": vae_lr})
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=_cfg(opt, "adam_weight_decay", 0.01),
            eps=_cfg(opt, "adam_epsilon", 1e-8),
        )
        self.optimizers.append(optimizer)

    def setup_dataloaders(self):
        # FullResPatchifierDataset is an IterableDataset that yields
        # already-batched tensors itself -- see VAEBlurAlign_model's
        # setup_dataloaders (duplicated here, not imported, so this file has
        # no import-order dependency on that class beyond lora_adapt_vae_encoder).
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
            prompt_embeds, text_ids = pipe.encode_prompt([""], num_images_per_prompt=1)
        self.register_buffer_txt = prompt_embeds.cpu()
        self.register_buffer_txt_ids = text_ids.cpu()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def _prepare_text_inputs(self, batch_size: int, device: torch.device):
        self._ensure_null_embeddings()
        txt = self.register_buffer_txt.to(device).expand(batch_size, -1, -1).to(self.weight_dtype)
        txt_ids = self.register_buffer_txt_ids.to(device).to(self.weight_dtype)
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return txt, txt_ids

    def _transformer_guidance(self, batch_size: int, device: torch.device):
        if not self.use_transformer_guidance:
            return None
        return torch.full((batch_size,), self.guidance_scale, device=device, dtype=self.weight_dtype)

    def _cond_image_latents(self, mixed_image: torch.Tensor, bsz: int):
        """Encode the conditioning ("mixed") image via cond_vae. Gradient
        flows through only when vae_trainable=True; otherwise this is a plain
        no_grad forward, matching cond_vae's requires_grad=False params."""
        if self.vae_trainable:
            return prepare_image_latents(
                vae=self.cond_vae, ref_images=mixed_image, batch_size=bsz,
                generator=None, device=self.accelerator.device, dtype=self.weight_dtype,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )
        with torch.no_grad():
            return prepare_image_latents(
                vae=self.cond_vae, ref_images=mixed_image, batch_size=bsz,
                generator=None, device=self.accelerator.device, dtype=self.weight_dtype,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )

    def optimize_parameters(self):
        self._ensure_null_embeddings()

        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        gt_image = self.sample[gt_key].clip(-1, 1)
        mixed_image = self.sample[lq_key].clip(-1, 1)
        bsz = gt_image.shape[0]

        with torch.inference_mode():
            pixel_latents = encode_vae_image(
                self.vae, gt_image.to(self.weight_dtype), None, flux2_pipe_cls=self.flux2_pipe_cls
            )
        image_latents, image_latent_ids = self._cond_image_latents(mixed_image.to(self.weight_dtype), bsz)

        noise = torch.randn_like(pixel_latents)
        noisy_latents, timesteps, _ = get_noisy_model_input_and_timesteps(
            self.noise_scheduler_copy, pixel_latents, noise, pixel_latents.device,
            discrete_flow_shift=self.discrete_flow_shift,
        )

        packed_transformer_input = self.flux2_pipe_cls._pack_latents(noisy_latents)
        transformer_image_ids = self.flux2_pipe_cls._prepare_latent_ids(noisy_latents).to(
            device=self.accelerator.device, dtype=self.weight_dtype
        )
        latent_token_count = packed_transformer_input.shape[1]

        txt, txt_ids = self._prepare_text_inputs(bsz, self.accelerator.device)
        transformer_guidance = self._transformer_guidance(bsz, self.accelerator.device)
        normalized_timesteps = (timesteps.float() / 1000).to(self.weight_dtype)

        # Conditioning tokens go into hidden_states alongside the noisy
        # target's own tokens (not encoder_hidden_states, which stays pure
        # null-text) -- image_latent_ids come from _prepare_image_ids, a
        # different id scheme than the target's _prepare_latent_ids, so the
        # two token sets don't collide positionally.
        latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
        latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

        model_pred = self.transformer(
            hidden_states=latent_model_input.to(self.weight_dtype),
            encoder_hidden_states=txt.to(self.weight_dtype),
            timestep=normalized_timesteps,
            guidance=transformer_guidance,
            txt_ids=txt_ids.to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]
        model_pred = model_pred[:, :latent_token_count]
        model_pred = self.flux2_pipe_cls._unpack_latents_with_ids(model_pred, transformer_image_ids)

        target = noise - pixel_latents
        loss = F.mse_loss(model_pred.float(), target.float())

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            grad_params = [p for p in self.transformer.parameters() if p.requires_grad]
            if self.vae_trainable:
                grad_params += [p for p in self.cond_vae.parameters() if p.requires_grad]
            self.accelerator.clip_grad_norm_(
                grad_params, getattr(self.opt.train.optim, "max_grad_norm", 1.0),
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
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            if idx >= self.max_val_steps:
                break

        gc.collect()
        torch.cuda.empty_cache()
        for model in self.models:
            model.train()

    @torch.no_grad()
    def forwardpass(self, mixed_image: torch.Tensor, height: int, width: int, num_steps: int = 50):
        self._ensure_null_embeddings()
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = mixed_image.shape[0]

        txt, txt_ids = self._prepare_text_inputs(bsz, device)

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_spatial_divisor = vae_scale_factor * 2
        lat_c = self.transformer.config.in_channels
        lat_h = int(height) // latent_spatial_divisor
        lat_w = int(width) // latent_spatial_divisor

        transformer_guidance = self._transformer_guidance(bsz, device)
        image_latents, image_latent_ids = prepare_image_latents(
            vae=self.cond_vae, ref_images=mixed_image.to(dtype), batch_size=bsz,
            generator=None, device=device, dtype=dtype, flux2_pipe_cls=self.flux2_pipe_cls,
        )

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

        latents = torch.randn(bsz, lat_c, lat_h, lat_w, device=device, dtype=dtype)

        for t in tqdm(scheduler.timesteps, desc="sampling", leave=False):
            packed_transformer_input = self.flux2_pipe_cls._pack_latents(latents)
            transformer_image_ids = self.flux2_pipe_cls._prepare_latent_ids(latents).to(device=device, dtype=dtype)
            latent_token_count = packed_transformer_input.shape[1]

            timestep = t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0
            latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
            latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=txt.to(dtype),
                timestep=timestep,
                guidance=transformer_guidance,
                txt_ids=txt_ids.to(dtype),
                img_ids=latent_image_ids.to(dtype),
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latent_token_count]
            noise_pred = self.flux2_pipe_cls._unpack_latents_with_ids(noise_pred, transformer_image_ids)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = decode_latents_flux(latents, self.vae, flux2_pipe_cls=self.flux2_pipe_cls)
        return decoded.clamp(-1, 1)

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = getattr(self.opt.val, "num_inference_steps", 50)

        mixed_image = self.sample[lq_key].clip(-1, 1)
        gt_image = self.sample[gt_key].clip(-1, 1)
        bsz = mixed_image.shape[0]

        out = self.forwardpass(mixed_image, mixed_image.shape[-2], mixed_image.shape[-1], num_inference_steps)

        mixed_np = (mixed_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
        gt_np = (gt_image.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        val_metrics = getattr(self.opt.val, "metrics", None)
        for i in range(bsz):
            idx += 1
            panels = np.stack([mixed_np[i], out_np[i], gt_np[i]])
            log_image(self.opt, self.accelerator, panels, f"{idx:04d}_mixed_pred_gt", self.global_step)
            if val_metrics:
                log_metrics(gt_np[i], out_np[i], val_metrics, self.accelerator, self.global_step)
                log_metrics(gt_np[i], mixed_np[i], val_metrics, self.accelerator, self.global_step, comment="floor_")
        return idx

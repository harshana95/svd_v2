"""Wan22VJEPADynamics_model: generate a short clip from a first frame + V-JEPA
"blur dynamics" tokens, injected into Wan 2.2 TI2V-5B's self-attention via a
PGA-style trainable adapter (see models/MotionBlur2Video/wan_dynamics_adapters/).

Staged, per the approved plan (see the plan file / findings for full
rationale): rather than building directly on top of the never-yet-trained
Phase-1 BlurJepaViT_arch (models/reFlux/BlurJepaDynamics_model.py), this
model supports two interchangeable dynamics sources, gated by
`dynamics_source`, so the Wan-injection architecture itself can be validated
in isolation first:

  - "vjepa_gt"   (Stage A, default): dynamics tokens = real frozen V-JEPA
                 output computed directly on the ground-truth clip itself.
                 An intentional oracle/upper-bound signal for validating
                 that the injection mechanism actually steers generation --
                 not the deployable model.
  - "blur_jepa"  (Stage B): dynamics tokens = BlurJepaViT_arch's prediction
                 from the single blurred frame alone (frozen, loaded from a
                 completed Phase-1 checkpoint). This is the actual
                 deployable model.

Both stages share every line of code downstream of `_dynamics_full_grid` --
only the source of the raw [B, vjepa_num_tubelets*grid_h*grid_w, feature_dim]
token grid differs.

Burst mode (`burst_size` > 1, default 1): conditions on a temporally-ordered
BURST of `burst_size` blurred images instead of one (Adobe240Dataset's
`burst_blur`/`burst_frames` keys), to (a) generate a proportionally longer
video and (b) give the model an input that is not time-symmetric the way a
single blur image is -- see dataset/adobe240_dataset.py's burst_size
docstring for the motivation. Each burst segment is run through the exact
same per-segment dynamics-source logic used for burst_size=1 (real V-JEPA on
that segment's own real sub-window for "vjepa_gt", or the frozen predictor
on that segment's blur image for "blur_jepa"), and the resulting per-segment
token grids are concatenated along the tubelet axis into one virtual long
clip (see `_dynamics_full_grid_burst`) -- every function downstream of that
concatenation (pooling/resampling/connector/broadcast/PGA-injection) is
completely unchanged from the burst_size=1 path. The deblur-priming prefix
(deblur_prefix_frames) is applied only ONCE, at the very start of the whole
burst (see `_first_frame_condition`), not per segment.

First-frame conditioning follows Wan2.2 TI2V-5B's own "expand_timesteps"
diffusion-forcing mechanism (verified directly against
diffusers/pipelines/wan/pipeline_wan_i2v.py's `prepare_latents`/`__call__`,
diffusers==0.39.0.dev0): the clean, VAE-encoded first frame is broadcast into
latent position 0 regardless of the sampled flow time, and a per-token
timestep tensor (shape [B, seq_len], one value per post-patch token) carries
timestep=0 at that frame's tokens and the real sampled t everywhere else --
TI2V-5B has no CLIP-image cross-attention path (`image_dim=None`) and no
channel-concat mask, unlike the Wan2.1-I2V-14B model
models/MotionBlur2Video/Blur2Vid_Wan_model.py already trains in this repo.
"""
import os

import cv2
import numpy as np
import ptlflow
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
from ptlflow.utils import flow_utils
from transformers import AutoTokenizer, UMT5EncoderModel
from tqdm import tqdm

from models.archs import define_network
from models.base_model import BaseModel
from models.LucidFlux.jepa_helpers import (
    extract_vjepa_tokens,
    extract_vjepa_tokens_layers,
    get_vjepa_feature_dim,
    load_vjepa2_model,
    load_vjepa2_weights,
    vjepa_from_unit_tensor,
)
from models.MotionBlur2Video.helpers import apply_lora_to_model
from models.MotionBlur2Video.wan_dynamics_adapters.features import DynamicsPredictorWrapper
from models.MotionBlur2Video.wan_dynamics_adapters import (
    JEPAFactoredConnector,
    MultiLayerDynamicsConnector,
    WanBlurCondEmbed,
    broadcast_frame_tokens_to_video_tokens,
    install_wan_dynamics_adapters,
    pool_dynamics_to_frame_tokens,
    resample_dynamics_grid,
)
from models.archs.BlurJepaViT_arch import BlurJepaViT_arch
from models.reFlux.blur_adapters import JEPAPerceptualEncoder
from utils.misc import calculate_psnr, calculate_ssim, log_image, log_metric, log_video


def _cfg(opt, key, default=None):
    value = opt.get(key, default) if hasattr(opt, "get") else getattr(opt, key, default)
    return default if value in (None, "~") else value


class Wan22VJEPADynamics_model(BaseModel):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        # V-JEPA hub weights use fp32 attention (same caveat as every other
        # V-JEPA caller in this repo).
        self.vjepa_dtype = torch.float32

        model_id = opt.pretrained_model_name_or_path
        revision = _cfg(opt, "revision", "main")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision)
        # low_mem_eval (opt-in): load the UMT5-XXL encoder directly in weight_dtype (bf16: ~11 GB host RAM instead of ~23 GB in fp32;
        # it is cast to weight_dtype on the GPU below anyway, so the numerics are identical) and drop it once the empty-prompt
        # embedding is cached (frees ~11 GB GPU). For running evals next to another job under a shared RAM / GPU budget.
        self.low_mem_eval = bool(_cfg(opt, "low_mem_eval", False))
        if self.low_mem_eval:
            self.text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", revision=revision,
                                                                 torch_dtype=weight_dtype)
        else:
            self.text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", revision=revision)
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=weight_dtype, revision=revision
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32, revision=revision
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", revision=revision
        )
        # The checkpoint's own scheduler is UniPC with flow_shift 5, but the Euler scheduler loaded above has NO flow_shift key and falls
        # back to shift=1.0. Runs trained with uniform t are self-consistent with shift 1 (default kept bit-for-bit); runs trained with a
        # shifted t distribution (t_sampling shifted/logit_normal, t_shift=S) MUST sample with the same shift: set inference_flow_shift: S.
        _ifs = _cfg(opt, "inference_flow_shift", None)
        if _ifs is not None:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config, shift=float(_ifs))
            logger.info("inference scheduler: FlowMatchEuler shift=%.2f", float(_ifs))

        if _cfg(opt, "enable_slicing", False):
            self.vae.enable_slicing()
        if _cfg(opt, "enable_tiling", False):
            self.vae.enable_tiling()

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        # Gradients still have to flow *through* every frozen block to reach
        # the injected PGA/connector/modulation parameters, so activation
        # memory is close to full fine-tuning even though the backbone's own
        # weights never update -- opt-in checkpointing (same convention as
        # ContextFluxBlurMap_model's transformer_gradient_checkpointing).
        if _cfg(opt, "transformer_gradient_checkpointing", False):
            self.transformer.enable_gradient_checkpointing()

        self.vae_scale_factor_spatial = getattr(self.vae.config, "scale_factor_spatial", 16)
        self.vae_scale_factor_temporal = getattr(self.vae.config, "scale_factor_temporal", 4)

        # Capture config values as plain attributes now, before any LoRA
        # wrapping -- a peft.PeftModel's `.config` proxy is not something to
        # rely on downstream, so every later access in this file goes
        # through these cached values instead of `self.transformer.config`.
        self.patch_size = tuple(self.transformer.config.patch_size)
        self.max_text_seq_length = getattr(self.transformer.config, "max_text_seq_length", 512)
        heads = self.transformer.config.num_attention_heads
        head_dim = self.transformer.config.attention_head_dim
        self.wan_inner_dim = heads * head_dim

        # PGA-style dynamics-injection adapters on every self-attention
        # block -- installed before any LoRA wrapping so it always operates
        # on the raw WanTransformer3DModel.
        self.pga_rank = int(_cfg(opt, "pga_rank", 8))
        # dynamics_injection: how the V-JEPA signal reaches Wan.
        #   "value_residual" (default, what every existing checkpoint trained
        #       with): PGA adds a per-token vector to self-attention's `value`.
        #       Each position gets the SAME additive signal for a given frame
        #       under the pooled connector.
        #   "cross_attention": each position instead queries the compact
        #       dynamics set, so positions can read different content. Only
        #       buys anything if that set has structure to attend over, i.e.
        #       pair it with dynamics_connector_type: factored -- with pooled
        #       the set is just T distinct vectors and this is near-degenerate.
        # Both are zero-init, so either mode resumes a checkpoint unperturbed.
        # Grid dims are needed by the cross-attention modules at install time,
        # and burst mode multiplies the tubelet count by burst_size (the
        # per-segment grids are concatenated along the tubelet axis).
        _vj_gh = int(_cfg(opt, "vjepa_grid_h", 24))
        _vj_gw = int(_cfg(opt, "vjepa_grid_w", 24))
        _per_seg_tub = int(_cfg(opt, "vjepa_num_tubelets", 5))
        _vj_tubelets = _per_seg_tub * max(int(_cfg(opt, "burst_size", 1)), 1)
        # dynamic_burst: the burst size varies per training step, so pos_t must
        # be allocated for the LARGEST burst it will ever see. Sized here, at
        # install time, because the module is built before any data is seen.
        _dyn = _cfg(opt, "dynamic_burst", None)
        _dyn_max = int(_dyn.get("max", 0)) if _dyn is not None else 0
        _vj_max_tubelets = _per_seg_tub * max(_dyn_max, int(_cfg(opt, "burst_size", 1)), 1)
        # Explicit override. REQUIRED by any config that LOADS a checkpoint
        # trained with a different max burst size -- notably inference configs
        # (Stage C, the held-out eval), which have no `dynamic_burst` block and
        # would otherwise size pos_t from their own burst_size. Getting this
        # wrong at inference is worse than at training: the positions are not
        # relearned, the model simply runs with a partly-random pos_t.
        _mt_override = _cfg(opt, "xattn_max_tubelets", None)
        if _mt_override is not None:
            _vj_max_tubelets = int(_mt_override)
        logger.info(
            "cross-attention pos_t will be allocated for max_tubelets=%d "
            "(per_segment=%d, max burst=%d)%s",
            _vj_max_tubelets, _per_seg_tub,
            _vj_max_tubelets // max(_per_seg_tub, 1),
            " [xattn_max_tubelets override]" if _mt_override is not None else "",
        )
        self.dynamics_injection = _cfg(opt, "dynamics_injection", "value_residual")
        if self.dynamics_injection not in ("value_residual", "cross_attention"):
            raise ValueError(f"Unknown dynamics_injection: {self.dynamics_injection!r}")
        self.dynamics_context, self.modulation, self.pga, self.dynamics_xattn = \
            install_wan_dynamics_adapters(
                self.transformer, rank=self.pga_rank,
                cross_attention=self.dynamics_injection == "cross_attention",
                xattn_rank=int(_cfg(opt, "xattn_rank", 64)),
                n_tubelets=_vj_tubelets, grid_h=_vj_gh, grid_w=_vj_gw,
                max_tubelets=_vj_max_tubelets,
            )
        # "factored": JEPAFactoredConnector + resample_dynamics_grid (spatial
        # structure preserved, default for fresh runs). "pooled": the
        # original JEPAPerceptualEncoder + pool_dynamics_to_frame_tokens
        # (global spatial average, one vector per frame) -- REQUIRED to
        # correctly resume/evaluate any checkpoint trained before the
        # "factored" pipeline existed (a checkpoint's saved module is keyed
        # by class name; loading a "pooled"-trained checkpoint under
        # "factored" silently leaves the connector randomly initialized
        # instead of erroring, see findings/Wan22VJEPADynamics_findings.md).
        self.dynamics_connector_type = _cfg(opt, "dynamics_connector_type", "factored")
        if self.dynamics_connector_type not in ("factored", "pooled"):
            raise ValueError(f"Unknown dynamics_connector_type: {self.dynamics_connector_type!r}")
        self.disable_dynamics_injection = bool(_cfg(opt, "disable_dynamics_injection", False))

        # Deblur-priming prefix: prepend `deblur_prefix_frames` copies of the
        # real clip's own first (sharp) frame before VAE-encoding, so the
        # model has a dedicated region whose target is always "the true
        # sharp first frame" -- separating "deblur the blurred I2V
        # condition" from "predict the subsequent real motion" instead of
        # asking one frozen/lightly-adapted first latent frame to do both.
        # Zero by default (fully additive, no behavior change for existing
        # configs). Must be a multiple of vae_scale_factor_temporal so the
        # extra pixel-frames map to a whole number of extra latent frames
        # (a non-multiple would silently produce the exact class of
        # odd-latent-dimension bug already fixed for H/W at 720px -- see
        # findings/Wan22VJEPADynamics_findings.md).
        self.deblur_prefix_frames = int(_cfg(opt, "deblur_prefix_frames", 0))
        # Re-blur consistency loss (off by default): at LOW-noise timesteps decode the predicted clean video x0_hat and require
        # mean_t(frames) ~ the input blur (the physical forward model; PickMoment's additivity idea). Needs a VAE decode WITH
        # grad (gradient-checkpointed) so it is only run on the samples with t < reblur_t_max.
        self.reblur_weight = float(_cfg(opt, "reblur_weight", 0.0))
        self.reblur_t_max = float(_cfg(opt, "reblur_t_max", 0.3))
        self.reblur_loss_type = str(_cfg(opt, "reblur_loss", "l1"))          # l1 | mse
        self.reblur_target = str(_cfg(opt, "reblur_target", "blur"))         # blur | gt_mean
        self.reblur_frames = int(_cfg(opt, "reblur_max_samples", 1))         # decode at most this many samples per step
        # The decode+backward through Wan's VAE is the memory hot spot (OOM'd an 80GB A100 on a 704px crop): skip the
        # re-blur term on crops larger than this many pixels (H*W). 0 = no cap.
        self.reblur_max_pixels = int(_cfg(opt, "reblur_max_pixels", 0))
        # Decoded-pixel loss on the CENTER generated frame vs the GT center frame (same decode as the re-blur term). The
        # center frame is invariant to time reversal, so unlike per-frame losses it does not punish a valid reversed clip.
        # pixel_loss: l1 | lpips | l1+lpips.  Needs `lpips` (+ torchvision vgg16 weights reachable) for the lpips variants.
        self.pixel_loss_weight = float(_cfg(opt, "pixel_loss_weight", 0.0))
        self.pixel_loss_type = str(_cfg(opt, "pixel_loss", "l1+lpips"))
        self.pixel_t_max = float(_cfg(opt, "pixel_t_max", self.reblur_t_max))
        self.pixel_max_pixels = int(_cfg(opt, "pixel_max_pixels", self.reblur_max_pixels))
        self._lpips_net = None
        # Timestep sampling for TRAINING. uniform (old behaviour) | shifted (t=shift*u/(1+(shift-1)u), the Wan/SD3 schedule
        # warp; match `t_shift` to the inference scheduler's flow_shift) | logit_normal (sigmoid(N(mean,std)), then the same shift).
        self.t_sampling = str(_cfg(opt, "t_sampling", "uniform"))
        if self.t_sampling not in ("uniform", "shifted", "logit_normal"):
            raise ValueError(f"unknown t_sampling {self.t_sampling!r}")
        self.t_shift = float(_cfg(opt, "t_shift", 1.0))
        self.t_logit_mean = float(_cfg(opt, "t_logit_mean", 0.0))
        self.t_logit_std = float(_cfg(opt, "t_logit_std", 1.0))
        # Low-noise stratum: this fraction of each batch's samples draws t ~ U(0, t_low_max) regardless of t_sampling. Needed when the
        # decoded-pixel losses (reblur/pixel, active only at t < ~0.3) are on together with a high-noise-heavy shifted schedule,
        # which would otherwise starve them (shift 5 puts <1% of draws below t=0.3).
        self.t_low_frac = float(_cfg(opt, "t_low_frac", 0.0))
        self.t_low_max = float(_cfg(opt, "t_low_max", 0.3))
        # Oracle->predicted robustness (TRAIN only, vjepa_gt oracle tokens): zero a random fraction of whole tokens and add
        # Gaussian noise of `token_noise_std` x the per-layer-slot token RMS, so Stage C's imperfect tokens are in-distribution.
        self.token_dropout = float(_cfg(opt, "token_dropout", 0.0))
        self.token_noise_std = float(_cfg(opt, "token_noise_std", 0.0))
        # Blur latent visible at EVERY latent frame (S=1 analogue of burst_blur_conditioning): zero-init WanBlurCondEmbed on the
        # clean blur latent tiled over time, added at block 0. Starts as an exact no-op.
        self.blur_every_frame = bool(_cfg(opt, "blur_every_frame_conditioning", False))
        # Trainable full-weight modules (unfreeze_modules) must be fp32 masters, else 1e-5-scale updates vanish in bf16.
        # `fp32_trainable: true` extends this to the LoRA A/B weights: a bf16 LoRA_A (|w|~0.02, ulp ~1e-4) silently drops Adam steps of
        # ~lr=1e-4. Off by default so existing configs/checkpoints reproduce bit-for-bit.
        self._fp32_lora = bool(_cfg(opt, "fp32_trainable", False))
        self._unfreeze_fp32 = bool(_cfg(opt, "unfreeze_modules", None)) or self._fp32_lora
        # Wan's VAE is CAUSAL: latent 0 absorbs exactly ONE pixel frame and every
        # later latent absorbs vae_scale_factor_temporal. So a prefix lands on a
        # latent boundary iff p % 4 == 1 (p in 1, 5, 9, ...), NOT p % 4 == 0.
        # The original guard here required p % 4 == 0, which accepts exactly the
        # straddling values and rejects exactly the aligned ones -- with the
        # historical p=4, latent 1 is 3/4 prefix + 1/4 real frame. Both
        # congruences are accepted now: `1` is the correct, aligned family, and
        # `0` is kept ONLY so every pre-existing checkpoint/config in this line
        # (all of which use p=4) still loads and reproduces. See
        # findings/Wan22VJEPADynamics_findings.md's guard entry.
        rem = self.deblur_prefix_frames % self.vae_scale_factor_temporal
        if self.deblur_prefix_frames != 0 and rem not in (0, 1):
            raise ValueError(
                f"deblur_prefix_frames={self.deblur_prefix_frames} must satisfy "
                f"p % {self.vae_scale_factor_temporal} == 1 (latent-aligned; preferred) or == 0 "
                f"(legacy, straddles a latent boundary -- kept only for checkpoint compatibility)."
            )
        self._prefix_latent_aligned = rem == 1
        # v3 REAL-context layout (replaces the copy-prefix): the generated video is
        #   [ctx_front real frames][wan_num_frames exposure frames][ctx_back real frames]
        # where the first ctx_front frames = 1 condition-slot frame (pixel 0 -> latent 0, replaced by the blur latent but it still
        # leaks into later latents through the causal VAE, so it is a REAL frame) + (ctx_front-1) real past frames. All context
        # frames are contiguous GT video from the dataset (`context_key`, e.g. 19 frames = 5 past | 9 exposure | 5 future).
        # Needs (ctx_front + wan_num_frames + ctx_back - 1) % 4 == 0 (e.g. 5 + 9 + 3 = 17 -> 5 latents). deblur_prefix_frames must be 0.
        self.context_key = _cfg(opt, "context_key", None)
        _lay = _cfg(opt, "context_layout", None)
        if self.context_key:
            self.ctx_front = int(_lay.get("front", 5)) if _lay is not None else 5
            self.ctx_back = int(_lay.get("back", 3)) if _lay is not None else 3
            if self.deblur_prefix_frames != 0:
                raise ValueError("context_key (v3 real-context layout) requires deblur_prefix_frames: 0")
            if self.dynamics_injection != "cross_attention":
                raise ValueError("context_key (v3) requires dynamics_injection=cross_attention")
        else:
            self.ctx_front, self.ctx_back = self.deblur_prefix_frames, 0
        # flow-matching weight on latents that contain ONLY context frames (the blur says little about them)
        self.ctx_loss_weight = float(_cfg(opt, "ctx_loss_weight", 0.5))

        self.lora_finetune = bool(_cfg(opt, "lora_finetune", False))
        if self.lora_finetune:
            self.transformer.enable_gradient_checkpointing()
            self.transformer = apply_lora_to_model(
                self.transformer,
                r=int(_cfg(opt, "lora_r", 8)),
                lora_alpha=int(_cfg(opt, "lora_alpha", 16)),
                lora_dropout=float(_cfg(opt, "lora_dropout", 0.0)),
                target_modules=list(_cfg(opt, "lora_target_modules", ["to_q", "to_k", "to_v", "to_out"])),
                # e.g. [patch_embedding, proj_out, condition_embedder, norm2]: trained in full alongside the LoRA.
                modules_to_save=list(_cfg(opt, "unfreeze_modules", []) or []),
            )

        # --- dynamics source (Stage A: vjepa_gt / Stage B: blur_jepa) ---
        self.dynamics_source = _cfg(opt, "dynamics_source", "vjepa_gt")
        self.vjepa_hub_entry = _cfg(opt, "vjepa_hub_entry", "vjepa2_1_vit_large_384")
        self.vjepa_image_size = int(_cfg(opt, "vjepa_image_size", 384))
        self.vjepa_num_tubelets = int(_cfg(opt, "vjepa_num_tubelets", 5))
        self.vjepa_grid_h = int(_cfg(opt, "vjepa_grid_h", 24))
        self.vjepa_grid_w = int(_cfg(opt, "vjepa_grid_w", 24))
        self.vjepa_feature_dim = get_vjepa_feature_dim(
            self.vjepa_hub_entry, fallback_dim=int(_cfg(opt, "vjepa_feature_dim", 1024))
        )
        # Multi-layer conditioning: `vjepa_layers` = 0-based V-JEPA block indices (e.g. [5, 11, 17, 23]; the last
        # block's slot is the final normed output). Empty/None = last output only (old behaviour, byte-identical).
        # The oracle grid is then [B, N, D*K]; MultiLayerDynamicsConnector turns it into K KV sets, and Wan block
        # group g (of K) cross-attends to set g. vjepa_feature_dim stays the PER-LAYER width D.
        self.vjepa_layers = [int(l) for l in (_cfg(opt, "vjepa_layers", None) or [])]
        self.vjepa_num_layers = max(len(self.vjepa_layers), 1)
        if len(self.vjepa_layers) > 1:
            if self.dynamics_injection != "cross_attention" or self.dynamics_source != "vjepa_gt":
                raise ValueError("vjepa_layers (K>1) requires dynamics_injection=cross_attention and dynamics_source=vjepa_gt (Stage A oracle)")

        # wan_num_frames must match between train/val (both used to derive
        # V-JEPA's expected even-frame-count input for the "vjepa_gt" stage
        # at init time; a per-batch mismatch is treated as a config error,
        # not silently handled per-batch).
        wan_num_frames_cfg = int(_cfg(opt.datasets.train, "wan_num_frames", 9))

        # burst_size > 1: condition on a temporally-ordered BURST of
        # burst_size blurred images (Adobe240Dataset's burst_key/
        # burst_frames_key) instead of one, to generate a proportionally
        # longer video AND give the model an input that is not
        # time-symmetric the way a single blur image is (see
        # dataset/adobe240_dataset.py's burst_size docstring and
        # findings/ for the full rationale). Purely additive: burst_size=1
        # (default) leaves every code path in this file byte-identical.
        self.burst_size = int(_cfg(opt, "burst_size", 1))
        # sampler_seed: seed for the DENOISING NOISE at validation/inference. Unset (default) keeps the
        # old behaviour bit-for-bit. WHY a separate key: `seed` is applied by accelerate.set_seed at
        # startup, but every eval RESUMES from a checkpoint and accelerator.load_state() then restores
        # the checkpoint's saved RNG state (random_states_0.pkl), silently overwriting `seed`. Five
        # "different seeds" therefore produced byte-identical videos (identical md5) and a noise floor
        # of exactly 0. A generator seeded HERE is applied after the resume and cannot be overwritten.
        _ss = _cfg(opt, "sampler_seed", None)
        self.sampler_seed = int(_ss) if _ss is not None else None

        # dynamics_encoder_type: which frozen encoder produces the ORACLE dynamics tokens (vjepa_gt path).
        # "vjepa" (default, unchanged) or "dinov2": DINOv2-large PER FRAME at 336x336 (patch 14 -> the same
        # 24x24=576 grid, dim 1024), adjacent frames pair-averaged to mimic V-JEPA's tubelet_size=2, so it drops
        # into the identical [B, 12*576, 1024] slots. It has NO temporal modelling of its own -- the point of
        # the P2c ablation is to test whether V-JEPA's temporal features are what carries the gain.
        self.dynamics_encoder_type = str(_cfg(opt, "dynamics_encoder_type", "vjepa"))
        if self.dynamics_encoder_type not in ("vjepa", "dinov2"):
            raise ValueError(f"unknown dynamics_encoder_type {self.dynamics_encoder_type!r}")
        self.dino = None

        # intervention_suite (eval only): run a list of ARMS x noise SEEDS inside ONE process so the 5B model
        # loads once. Each arm perturbs either the dynamics TOKENS (mode: zero /
        # reverse_within_seg / shift_segments / shuffle_tubelets / shuffle_space / noise / keep_seg /
        # swap_donor) or a MODULE (xattn_off / lora_off / pos_t_shuffle / pos_s_shuffle / pos_zero), and writes
        # its videos to <run>/<arm>_s<seed>/videos so the existing scorers work on it unchanged.
        # NB config values are DictAsMember: only .get() is supported (see dictasmember_silent_empty_dict).
        self.suite = None
        self.active_arm = None
        self._arm_undo = []
        self._lora_ctx = None
        self._clip_stems = []
        self._donor_log = []
        self._donor_viz = None  # set by _apply_token_arm's swap_donor case; read+cleared by validate_step
        self._val_stem_suffix = ""  # set by validation() when dynamic_burst.val_sizes has >1 entry
        _su = _cfg(opt, "intervention_suite", None)
        if _su is not None and bool(_su.get("enabled", False)):
            _arms = []
            for a in list(_su.get("arms")):
                _arms.append({k: a.get(k, None) for k in ("name", "mode", "seg", "sigma", "offset", "scale", "control", "alpha", "target")})
            self.suite = {"arms": _arms, "seeds": [int(x) for x in list(_su.get("seeds", [1]))]}
            logger.info("intervention_suite: %d arms x seeds %s", len(_arms), self.suite["seeds"])

        # --- dynamic_burst: vary the burst size per training step ---------
        # The datasets emit the MAXIMUM burst; feed_data truncates to a
        # per-step S. That works because segment i occupies burst_frames[i]
        # and full-span frames [stride*i, stride*i + window_size), so the
        # first S segments are themselves a valid burst-S sample, and the
        # matching GT video is the first F_S = F_max - stride*(S_max - S)
        # frames of wan_frames (verified empirically: segment i's first frame
        # sits at wan_frames offset stride*i - 1 with zero residual).
        #
        # Cross-attention makes this nearly free: its KV is a SET, so its
        # length need not match Wan's token count. The only parameter that
        # was pinned to the tubelet count is WanDynamicsCrossAttn.pos_t, now
        # allocated at max_tubelets and sliced per forward.
        _dyn = _cfg(opt, "dynamic_burst", None)
        self.dyn_burst = None
        if _dyn is not None and bool(_dyn.get("enabled", False)):
            self.dyn_burst = {
                "min": int(_dyn.get("min", 1)),
                "max": int(_dyn.get("max", self.burst_size)),
                "ramp_steps": int(_dyn.get("ramp_steps", 0)),
                # Absolute step the ramp is measured FROM. Must be set to the
                # resume step on a warm start: the schedule uses global_step,
                # which on resume already exceeds ramp_steps, so a ramp left at
                # 0 silently collapses to "full range immediately" and the
                # curriculum never happens.
                "ramp_start_step": int(_dyn.get("ramp_start_step", 0)),
                "weight_power": float(_dyn.get("weight_power", 1.0)),
                "center_stride": int(_dyn.get("center_stride", 8)),
                "val_size": int(_dyn.get("val_size", 0)) or None,
                "seed": int(_dyn.get("seed", 1234)),
                # Training-time validation used to be pinned at a single size (val_size or max) --
                # "validation never caught [a bug] because val is pinned at S=max" (see the
                # pool_dynamics_to_frame_tokens comment below). val_sizes lets a training run's
                # periodic validation ALSO log burst=3 and burst=5 (etc.), not just one size, so
                # regressions specific to a burst length show up during training, not only in a
                # separate sweep after the fact. Defaults to the old single-value behavior.
                "val_sizes": (
                    [int(x) for x in _dyn["val_sizes"]] if _dyn.get("val_sizes", None) is not None
                    else [int(_dyn.get("val_size", 0)) or self.burst_size]
                ),
            }
            d = self.dyn_burst
            if not 1 <= d["min"] <= d["max"]:
                raise ValueError(f"dynamic_burst min/max invalid: {d['min']}..{d['max']}")
            if d["max"] != self.burst_size:
                raise ValueError(
                    f"dynamic_burst.max ({d['max']}) must equal burst_size ({self.burst_size}), "
                    "which is the size the DATASETS emit and everything is truncated from."
                )
            # Every reachable burst size must still satisfy Wan's frame-count
            # rule, (F + prefix - 1) % 4 == 0. Checking here turns a mid-run
            # crash at a rare sampled S into a startup error.
            f_max = int(_cfg(opt.datasets.train, "wan_num_frames", 0)) or int(
                _cfg(opt.datasets.train.sources[0], "wan_num_frames", 25))
            pre = int(_cfg(opt, "deblur_prefix_frames", 0) or 0)
            bad = [(S, f_max - d["center_stride"] * (d["max"] - S))
                   for S in range(d["min"], d["max"] + 1)
                   if (f_max - d["center_stride"] * (d["max"] - S) + pre - 1) % 4 != 0]
            if bad:
                raise ValueError(
                    f"dynamic_burst: these burst sizes give an invalid frame count "
                    f"(need (F + {pre} - 1) %% 4 == 0): {bad}. Adjust wan_num_frames "
                    f"(={f_max}), center_stride (={d['center_stride']}) or the S range."
                )
            if _cfg(opt, "dynamics_injection", "value_residual") != "cross_attention":
                raise ValueError(
                    "dynamic_burst requires dynamics_injection: cross_attention. The "
                    "value_residual path pools/interpolates the grid to a FIXED "
                    "vjepa_num_tubelets, so a varying burst size cannot work there."
                )
            bad_val_sizes = [v for v in d["val_sizes"] if not (d["min"] <= v <= d["max"])]
            if bad_val_sizes:
                raise ValueError(
                    f"dynamic_burst.val_sizes {bad_val_sizes} outside [{d['min']},{d['max']}]"
                )
            logger.info(
                "dynamic_burst ENABLED: S in [%d,%d], ramp over %d steps, "
                "P(S) ~ S^-%.2f, val at S=%s; F_S = %d - %d*(%d - S)",
                d["min"], d["max"], d["ramp_steps"], d["weight_power"],
                d["val_sizes"], f_max, d["center_stride"], d["max"],
            )

        if self.burst_size > 1:
            dataset_burst_size = int(_cfg(opt.datasets.train, "burst_size", 1))
            if dataset_burst_size != self.burst_size:
                raise ValueError(
                    f"model burst_size={self.burst_size} must match "
                    f"datasets.train.burst_size={dataset_burst_size}."
                )

        self.dynamics_encoder = None
        self.vjepa = None
        # P2g joint fine-tune: let Wan's own loss adapt the Stage-B predictor (its LoRA becomes trainable and is
        # handed to the optimizer). predictor_aux_weight > 0 adds an ANCHOR: MSE(predicted tokens, oracle V-JEPA
        # tokens) so the predictor cannot drift arbitrarily. Both default off -> the frozen-predictor behaviour is unchanged.
        self.predictor_trainable = bool(_cfg(opt, "dynamics_predictor_trainable", False))
        self.predictor_aux_weight = float(_cfg(opt, "predictor_aux_weight", 0.0))
        # "from_scratch" (default): BlurJepaViT_arch, a small from-scratch
        # ViT (models/reFlux/BlurJepaDynamics_model.py's original predictor).
        # "vjepa_lora": a second V-JEPA instance LoRA-adapted to predict its
        # own tokens from a repeated-frame blur "clip" instead -- see that
        # same model file's newer predictor_type=vjepa_lora path (converges
        # far faster since it starts from V-JEPA's own pretrained weights
        # instead of random init; see
        # findings/Wan22VJEPADynamics_findings.md's 2026-09-08 entry).
        self.dynamics_predictor_type = _cfg(opt, "dynamics_predictor_type", "from_scratch")
        if self.dynamics_predictor_type not in ("from_scratch", "vjepa_lora"):
            raise ValueError(f"Unknown dynamics_predictor_type: {self.dynamics_predictor_type!r}")

        if self.dynamics_source == "blur_jepa":
            checkpoint_path = _cfg(opt, "dynamics_encoder_checkpoint", None)
            if self.dynamics_predictor_type == "vjepa_lora":
                if not checkpoint_path:
                    raise ValueError(
                        "dynamics_predictor_type='vjepa_lora' requires dynamics_encoder_checkpoint "
                        "(the saved PEFT adapter directory, e.g. .../checkpoint-N/PeftModel_1)."
                    )
                from peft import PeftModel

                logger.info(
                    "Loading a LoRA-adapted V-JEPA dynamics predictor from %s", checkpoint_path
                )
                dynamics_base = load_vjepa2_model(
                    self.vjepa_hub_entry, "cpu", self.vjepa_dtype, offload=False,
                    pretrained=bool(_cfg(opt, "vjepa_pretrained", True)),
                )
                self.dynamics_encoder = PeftModel.from_pretrained(
                    dynamics_base, checkpoint_path, is_trainable=self.predictor_trainable)
                if self.predictor_trainable:
                    # LoRA params stay trainable (base V-JEPA weights are frozen by PEFT); unique class name, see wrapper
                    self.dynamics_encoder = DynamicsPredictorWrapper(self.dynamics_encoder)
                else:
                    self.dynamics_encoder.requires_grad_(False)
                # fp32 throughout, not weight_dtype -- V-JEPA's hub weights
                # use fp32 attention internally; bf16 inputs cause Q/K/V
                # dtype mismatches (same caveat documented on self.vjepa_dtype
                # above, and in BlurJepaDynamics_model.py for this identical
                # architecture).
                self.dynamics_encoder = self.dynamics_encoder.to(self.accelerator.device, dtype=self.vjepa_dtype)
                self.dynamics_encoder.eval()
                # OPTIONAL spatial motion cue (Stage B BlurJepaCueDynamics_model). Absent config -> nothing attached, behaviour unchanged.
                self.dynamics_cue = None
                self.dynamics_cue_mode = str(_cfg(opt, "dynamics_cue_mode", "none"))   # none | gt_flow | gt_flow_flip
                _cue_ckpt = _cfg(opt, "dynamics_cue_checkpoint", None)
                if _cue_ckpt:
                    from models.reFlux.motion_cue import MotionCueEncoder, attach_cue
                    self.dynamics_cue = MotionCueEncoder(int(_cfg(opt, "vjepa_feature_dim", 1024)), int(_cfg(opt, "dynamics_cue_hidden", 256)),
                                                         self.vjepa_grid_h).to(self.accelerator.device)
                    _sd = torch.load(os.path.join(_cue_ckpt, "pytorch_model.bin"), map_location="cpu")
                    self.dynamics_cue.load_state_dict(_sd)   # strict: a silent partial load would look like a working cue
                    self.dynamics_cue.requires_grad_(False)
                    attach_cue(self.dynamics_encoder, self.dynamics_cue)
                    logger.info("loaded motion-cue module from %s (dynamics_cue_mode=%s)", _cue_ckpt, self.dynamics_cue_mode)
                elif self.dynamics_cue_mode != "none":
                    raise ValueError("dynamics_cue_mode != none requires dynamics_cue_checkpoint")
                # OPTIONAL post-hoc token motion editor (models/reFlux/token_editor.py): edits the predicted tokens from a
                # user control map. Absent config -> nothing loaded. dynamics_edit_control: none | gt_fwd | gt_rev | json:<path>
                self.dynamics_editor = None
                self.dynamics_edit_control = str(_cfg(opt, "dynamics_edit_control", "none"))
                _ed_ckpt = _cfg(opt, "dynamics_editor_checkpoint", None)
                if _ed_ckpt:
                    from models.reFlux.token_editor import TokenMotionEditor
                    self.dynamics_editor = TokenMotionEditor(
                        int(_cfg(opt, "vjepa_feature_dim", 1024)), int(_cfg(opt, "dynamics_editor_dim", 256)),
                        int(_cfg(opt, "dynamics_editor_depth", 4)), int(_cfg(opt, "dynamics_editor_heads", 4)),
                        self.vjepa_grid_h).to(self.accelerator.device)
                    self.dynamics_editor.load_state_dict(torch.load(os.path.join(_ed_ckpt, "pytorch_model.bin"), map_location="cpu"))  # strict
                    self.dynamics_editor.requires_grad_(False)
                    self.dynamics_editor.eval()
                    logger.info("loaded token motion editor from %s (dynamics_edit_control=%s)", _ed_ckpt, self.dynamics_edit_control)
                elif self.dynamics_edit_control != "none":
                    raise ValueError("dynamics_edit_control != none requires dynamics_editor_checkpoint")
                # OPTIONAL user-stroke control (Stage B BlurJepaStrokeControl_model). Absent config -> nothing attached.
                # dynamics_stroke_mode (or suite arms stroke_fwd / stroke_rev / stroke_shuffle / stroke_none):
                # strokes synthesized from the GT clip (wan_frames[:, :8]) forward / reversed / taken from the previous clip.
                self.dynamics_stroke = None
                self.dynamics_stroke_mode = str(_cfg(opt, "dynamics_stroke_mode", "none"))   # none | gt | gt_rev | gt_shuffle
                self._prev_wan_strokes = None
                _st_ckpt = _cfg(opt, "dynamics_stroke_checkpoint", None)
                if _st_ckpt:
                    from models.reFlux import stroke_control as _sc
                    _mid = _cfg(opt, "dynamics_stroke_mid_block", 7)
                    self.dynamics_stroke = _sc.StrokeControlEncoder(
                        int(_cfg(opt, "vjepa_feature_dim", 1024)), int(_cfg(opt, "dynamics_stroke_hidden", 256)),
                        self.vjepa_grid_h, n_inject=1 if _mid is None else 2).to(self.accelerator.device)
                    _p = os.path.join(_st_ckpt, "pytorch_model.bin")
                    if not os.path.isfile(_p):
                        raise FileNotFoundError(_p)
                    self.dynamics_stroke.load_state_dict(torch.load(_p, map_location="cpu"))   # strict
                    if self.dynamics_stroke.last_layer_norm() == 0.0:
                        raise RuntimeError(f"{_p}: stroke control heads are at zero-init (would silently do nothing)")
                    self.dynamics_stroke.requires_grad_(False)
                    _sc.attach_stroke_control(self.dynamics_encoder, self.dynamics_stroke, _mid)
                    self.dynamics_stroke_cfg = _sc.StrokeCfg.from_opt(_cfg(opt, "dynamics_stroke_extract", None))
                    # must match the renderer the Stage B checkpoint was trained with (phase 2: soft band; None = phase 1)
                    self.dynamics_stroke_band = _sc.stroke_band_from_opt(_cfg(opt, "dynamics_stroke_band", None))
                    self.dynamics_stroke_raft = _sc.load_raft(self.accelerator.device)
                    logger.info("loaded stroke control from %s (mode=%s, |heads|=%.3f)", _st_ckpt, self.dynamics_stroke_mode,
                                self.dynamics_stroke.last_layer_norm())
                elif self.dynamics_stroke_mode != "none":
                    raise ValueError("dynamics_stroke_mode != none requires dynamics_stroke_checkpoint")
                # The checkpoint's own token-grid shape isn't recoverable
                # from a PEFT-wrapped raw nn.Module the way BlurJepaViT_arch's
                # HF .config exposes it -- these must match whatever
                # vjepa_window_size the BlurJepaViT_stageB_vjepalora.yml run
                # that produced this checkpoint actually used.
                self.dynamics_vjepa_repeat_frames = int(_cfg(opt, "dynamics_vjepa_repeat_frames", 10))
                # How the burst is handed to V-JEPA in blur_jepa (Stage C) mode.
                # MUST match how the Stage B predictor being loaded was TRAINED
                # (BlurJepaDynamics_model's `burst_vjepa_input`):
                #   "per_segment" (default, legacy): each segment is repeated
                #       into its OWN clip and encoded by a separate V-JEPA
                #       forward, then the grids are concatenated.
                #   "single_clip": the segments are repeated and concatenated
                #       into ONE temporally-ordered clip, encoded in a SINGLE
                #       forward -- mirrors _predict_tokens_burst_single_clip.
                # Both yield the same tubelet COUNT, so a mismatch does not
                # raise -- it silently feeds the predictor an input structure it
                # never saw, defeating the point of single_clip (which exists so
                # the encoder has cross-segment order to read).
                self.dynamics_burst_vjepa_input = _cfg(opt, "dynamics_burst_vjepa_input", "per_segment")
                if self.dynamics_burst_vjepa_input not in ("per_segment", "single_clip"):
                    raise ValueError(
                        f"Unknown dynamics_burst_vjepa_input: {self.dynamics_burst_vjepa_input!r}"
                    )
                self.vjepa_num_tubelets = int(_cfg(opt, "vjepa_num_tubelets", 5))
                self.vjepa_grid_h = int(_cfg(opt, "vjepa_grid_h", 24))
                self.vjepa_grid_w = int(_cfg(opt, "vjepa_grid_w", 24))
                self.vjepa_feature_dim = get_vjepa_feature_dim(
                    self.vjepa_hub_entry, fallback_dim=int(_cfg(opt, "vjepa_feature_dim", 1024))
                )
            elif checkpoint_path:
                logger.info("Loading frozen BlurJepaViT_arch dynamics predictor from %s", checkpoint_path)
                self.dynamics_encoder = BlurJepaViT_arch.from_pretrained(checkpoint_path)
                self.dynamics_encoder.requires_grad_(False)
                self.dynamics_encoder = self.dynamics_encoder.to(self.accelerator.device, dtype=weight_dtype)
                self.dynamics_encoder.eval()
                # Authoritative: read the token-grid shape from the loaded
                # arch's own config instead of trusting separately-configured
                # top-level vjepa_* opt keys, which could silently drift out
                # of sync with whatever BlurJepaDynamics.yml actually trained.
                enc_cfg = self.dynamics_encoder.config
                self.vjepa_num_tubelets = enc_cfg.vjepa_num_tubelets
                self.vjepa_grid_h = enc_cfg.vjepa_grid_h
                self.vjepa_grid_w = enc_cfg.vjepa_grid_w
                self.vjepa_feature_dim = enc_cfg.vjepa_feature_dim
            else:
                logger.warning(
                    "dynamics_encoder_checkpoint is unset -- using a randomly-initialized "
                    "BlurJepaViT_arch. This is a shape/wiring smoke-test only, not a real "
                    "Stage-B run."
                )
                self.dynamics_encoder = define_network(opt.dynamics_network)
                self.dynamics_encoder.requires_grad_(False)
                self.dynamics_encoder = self.dynamics_encoder.to(self.accelerator.device, dtype=weight_dtype)
                self.dynamics_encoder.eval()
                enc_cfg = self.dynamics_encoder.config
                self.vjepa_num_tubelets = enc_cfg.vjepa_num_tubelets
                self.vjepa_grid_h = enc_cfg.vjepa_grid_h
                self.vjepa_grid_w = enc_cfg.vjepa_grid_w
                self.vjepa_feature_dim = enc_cfg.vjepa_feature_dim
        elif self.dynamics_source == "vjepa_gt":
            logger.info("Loading frozen V-JEPA (Stage A ground-truth dynamics source): %s", self.vjepa_hub_entry)
            self.vjepa = load_vjepa2_model(
                self.vjepa_hub_entry, "cpu", self.vjepa_dtype, offload=False,
                pretrained=bool(_cfg(opt, "vjepa_pretrained", True)),
            )
            vjepa_checkpoint_path = _cfg(opt, "vjepa_checkpoint_path", None)
            if vjepa_checkpoint_path:
                load_vjepa2_weights(self.vjepa, vjepa_checkpoint_path)
            self.vjepa.requires_grad_(False)
            self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
            self.vjepa.eval()

            # Stage A's V-JEPA input frame count (`wan_num_frames`, forced
            # even) is unrelated to Phase-1's own vjepa_window_size, so the
            # tubelet count must be derived empirically for *this* config
            # rather than reusing a vjepa_num_tubelets default calibrated for
            # a different frame count (10, in BlurJepaDynamics.yml).
            #
            # Burst mode (advisor-reviewed): V-JEPA runs on each burst
            # SEGMENT's own vjepa_window_size real sub-window independently
            # (see _dynamics_full_grid_burst), not on the model's total
            # wan_num_frames -- so the empirical check here must use the
            # per-segment window size, then the resulting per-segment
            # tubelet count is multiplied by burst_size below to get the
            # total the downstream connector pipeline actually sees. Using
            # wan_num_frames_cfg here instead would derive a whole-clip
            # oracle layout that Stage 3's per-segment predictor could never
            # reproduce (findings/: "per-segment-then-concatenated, not
            # whole-clip").
            if self.burst_size > 1:
                burst_vjepa_window_size = int(_cfg(opt.datasets.train, "vjepa_window_size", 10))
                even_t = (burst_vjepa_window_size // 2) * 2
            else:
                even_t = (wan_num_frames_cfg // 2) * 2
            with torch.no_grad():
                dummy_clip = torch.zeros(
                    1, 3, even_t, self.vjepa_image_size, self.vjepa_image_size,
                    device=self.accelerator.device, dtype=self.vjepa_dtype,
                )
                dummy_tokens = self._vj_extract(dummy_clip)
            actual_n, actual_d = dummy_tokens.shape[1], dummy_tokens.shape[2]
            if actual_n % (self.vjepa_grid_h * self.vjepa_grid_w) != 0 or actual_d != self.vjepa_feature_dim * self.vjepa_num_layers:
                raise ValueError(
                    f"V-JEPA's actual output for a {even_t}-frame clip at vjepa_image_size="
                    f"{self.vjepa_image_size} is N={actual_n}, D={actual_d}, which doesn't factor "
                    f"cleanly as vjepa_grid_h*vjepa_grid_w={self.vjepa_grid_h * self.vjepa_grid_w} "
                    f"tokens/tubelet with D=vjepa_feature_dim={self.vjepa_feature_dim}. Update "
                    f"vjepa_grid_h/vjepa_grid_w/vjepa_feature_dim to match."
                )
            self.vjepa_num_tubelets = actual_n // (self.vjepa_grid_h * self.vjepa_grid_w)
            logger.info(
                "Stage A V-JEPA output shape verified for wan_num_frames=%d (even_t=%d): "
                "[B, %d, %d] = %d tubelets x %dx%d spatial",
                wan_num_frames_cfg, even_t, actual_n, actual_d,
                self.vjepa_num_tubelets, self.vjepa_grid_h, self.vjepa_grid_w,
            )
        else:
            raise ValueError(f"Unknown dynamics_source: {self.dynamics_source!r}")
        if self.dynamics_source == "blur_jepa" and (self.predictor_aux_weight > 0 or bool(_cfg(opt, "load_oracle_vjepa", False))) and self.vjepa is None:
            # aux anchor needs the ORACLE tokens too: load the frozen V-JEPA alongside the predictor
            logger.info("Loading frozen V-JEPA as the ORACLE for the predictor aux loss (weight %.3f)", self.predictor_aux_weight)
            self.vjepa = load_vjepa2_model(
                self.vjepa_hub_entry, "cpu", self.vjepa_dtype, offload=False,
                pretrained=bool(_cfg(opt, "vjepa_pretrained", True)),
            )
            self.vjepa.requires_grad_(False)
            self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
            self.vjepa.eval()
        # Neither self.dynamics_encoder nor self.vjepa is appended to
        # self.models -- both are frozen, exactly like
        # BlurJepaDynamics_model.vjepa / ContextFluxBlurMap_model.vjepa.

        # Burst mode: whichever branch above ran ("blur_jepa" or "vjepa_gt")
        # always derived a PER-SEGMENT tubelet count -- both the frozen
        # single-image predictors and the vjepa_gt empirical check above now
        # operate on one segment at a time. _dynamics_full_grid_burst
        # concatenates burst_size such per-segment grids along the tubelet
        # axis into one virtual long clip, so every downstream shape
        # (the startup assertion below, pool_dynamics_to_frame_tokens,
        # resample_dynamics_grid) must be told the TOTAL tubelet count.
        self.per_segment_vjepa_num_tubelets = self.vjepa_num_tubelets
        if self.burst_size > 1:
            self.vjepa_num_tubelets = self.per_segment_vjepa_num_tubelets * self.burst_size
            logger.info(
                "Burst mode: burst_size=%d x per_segment_vjepa_num_tubelets=%d = "
                "vjepa_num_tubelets=%d (total, concatenated)",
                self.burst_size, self.per_segment_vjepa_num_tubelets, self.vjepa_num_tubelets,
            )

        if self.vjepa_num_layers > 1:
            self.dynamics_connector = MultiLayerDynamicsConnector(self.vjepa_feature_dim, self.wan_inner_dim, self.vjepa_num_layers)
        elif self.dynamics_connector_type == "factored":
            self.dynamics_connector = JEPAFactoredConnector(self.vjepa_feature_dim, heads, head_dim)
        else:
            self.dynamics_connector = JEPAPerceptualEncoder(self.vjepa_feature_dim, self.wan_inner_dim)

        # Burst-blur conditioning: give Wan ALL burst segments as input, each
        # repeated over the frames it was averaged from, instead of only
        # segment 0 as a single first-frame I2V condition. Off by default, so
        # every existing config/checkpoint is unaffected; when on, the embedder
        # is zero-init so step 0 still reproduces the resumed checkpoint exactly.
        self.burst_blur_conditioning = bool(_cfg(opt, "burst_blur_conditioning", False))
        # Optional: cap the burst blurs' effective resolution (longer side) before VAE encoding; None = full crop.
        self.burst_blur_cond_size = _cfg(opt, "burst_blur_cond_size", None)
        self.burst_blur_cond_size = int(self.burst_blur_cond_size) if self.burst_blur_cond_size else None
        self.blur_cond_embed = None
        if self.burst_blur_conditioning and self.blur_every_frame:
            raise ValueError("burst_blur_conditioning and blur_every_frame_conditioning are mutually exclusive")
        if self.blur_every_frame and self.burst_size != 1:
            raise ValueError("blur_every_frame_conditioning is the S=1 variant (burst_size must be 1)")
        if self.burst_blur_conditioning or self.blur_every_frame:
            if self.burst_blur_conditioning and self.burst_size <= 1:
                raise ValueError("burst_blur_conditioning requires burst_size > 1")
            self.blur_cond_embed = WanBlurCondEmbed(int(self.vae.config.z_dim), self.wan_inner_dim, self.patch_size)
            self.blur_cond_embed = self.blur_cond_embed.to(self.accelerator.device, dtype=torch.float32)

        if self.lora_finetune:
            self.models.append(self.transformer)
        self.models.append(self.dynamics_connector)
        self.models.append(self.modulation)
        self.models.append(self.pga)
        if self.dynamics_xattn is not None:
            self.models.append(self.dynamics_xattn)
        if self.blur_cond_embed is not None:
            self.models.append(self.blur_cond_embed)
        if self.predictor_trainable and self.dynamics_encoder is not None:
            self.models.append(self.dynamics_encoder)

        self.transformer = self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        if self._unfreeze_fp32:
            n_fp32 = 0
            for n_, p_ in self.transformer.named_parameters():
                if p_.requires_grad and ("modules_to_save" in n_ or (self._fp32_lora and "lora_" in n_)):
                    p_.data = p_.data.float()
                    n_fp32 += 1
            logger.info("fp32 masters: %d trainable transformer tensors held in fp32 (forward runs under bf16 autocast; lora=%s)",
                        n_fp32, self._fp32_lora)
        self.text_encoder = self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=torch.float32)
        # Keep adapters in fp32 (matches blur_adapters' convention); they
        # cast back to block dtype where needed.
        self.dynamics_connector = self.dynamics_connector.to(self.accelerator.device, dtype=torch.float32)
        self.modulation = self.modulation.to(self.accelerator.device, dtype=torch.float32)
        self.pga = self.pga.to(self.accelerator.device, dtype=torch.float32)
        if self.dynamics_xattn is not None:
            self.dynamics_xattn = self.dynamics_xattn.to(self.accelerator.device, dtype=torch.float32)

        z_dim = self.vae.config.z_dim
        self.latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(self.accelerator.device)
        )
        self.latents_std = (
            (1.0 / torch.tensor(self.vae.config.latents_std)).view(1, z_dim, 1, 1, 1).to(self.accelerator.device)
        )

        self._empty_prompt_embeds = None

        # Frozen optical-flow model, used only for validation-time flow
        # visualization (matches Blur2Vid_Wan_with_OpticalFlow_model.py's
        # existing sea_raft usage in this repo).
        self.opt_flow_model = ptlflow.get_model("sea_raft_l", ckpt_path="things")
        self.opt_flow_model.eval()
        self.opt_flow_model.requires_grad_(False)
        self.opt_flow_model = self.opt_flow_model.to(self.accelerator.device, dtype=torch.float32)

        # Startup sanity check: verify the dynamics-token pipeline's output
        # sequence length actually matches Wan's real patchified token count
        # for the configured training resolution/frame-count (via a real
        # forward through the frozen patch_embedding, not just arithmetic),
        # mirroring BlurJepaDynamics_model.__init__'s analogous check.
        crop_size = int(_cfg(opt.datasets.train, "crop_size", 384))
        resize_mode = _cfg(opt.datasets.train, "resize_mode", "crop")
        if resize_mode == "resize":
            check_h = int(_cfg(opt.datasets.train, "resize_h", crop_size))
            check_w = int(_cfg(opt.datasets.train, "resize_w", crop_size))
        else:
            check_h = check_w = crop_size
        check_total_frames = wan_num_frames_cfg + self.ctx_front + self.ctx_back
        if (check_total_frames - 1) % self.vae_scale_factor_temporal != 0:
            raise ValueError(f"total generated frames {check_total_frames} (= {self.ctx_front} front + {wan_num_frames_cfg} exposure + "
                             f"{self.ctx_back} back) must be 1 mod {self.vae_scale_factor_temporal}")
        with torch.no_grad():
            t_lat_check = (check_total_frames - 1) // self.vae_scale_factor_temporal + 1
            h_lat_check = check_h // self.vae_scale_factor_spatial
            w_lat_check = check_w // self.vae_scale_factor_spatial
            dummy_latent = torch.zeros(
                1, z_dim, t_lat_check, h_lat_check, w_lat_check,
                device=self.accelerator.device, dtype=weight_dtype,
            )
            with self._maybe_autocast():
                dummy_patched = self.transformer.patch_embedding(dummy_latent)
            _, _, post_t, post_h, post_w = dummy_patched.shape
            expected_seq_len = post_t * post_h * post_w

            dummy_grid = torch.zeros(
                1, self.vjepa_num_tubelets * self.vjepa_grid_h * self.vjepa_grid_w, self.vjepa_feature_dim * self.vjepa_num_layers,
                device=self.accelerator.device, dtype=torch.float32,
            )
            if self.vjepa_num_layers > 1:
                # multi-layer cross-attention KV is a SET (list of K tensors); the value-residual seq-len check does not apply
                dummy_full = torch.zeros(1, expected_seq_len, 1)
            else:
                dummy_full = self._dynamics_tokens_with_deblur_padding(dummy_grid, post_t, post_h, post_w)
        if dummy_full.shape[1] != expected_seq_len:
            raise ValueError(
                f"Dynamics-token pipeline produces seq_len={dummy_full.shape[1]} but Wan's "
                f"patch_embedding produces {expected_seq_len} tokens for resolution={check_h}x{check_w}, "
                f"wan_num_frames={wan_num_frames_cfg}+deblur_prefix_frames={self.deblur_prefix_frames} "
                f"(post-patch shape ({post_t},{post_h},{post_w})). This should be impossible by "
                f"construction -- check vae_scale_factor_spatial/temporal and the transformer's patch_size."
            )
        logger.info(
            "Dynamics-token shape verified: seq_len=%d matches Wan's patchified token count "
            "for resolution=%dx%d, wan_num_frames=%d+deblur_prefix_frames=%d (dynamics_source=%s)",
            expected_seq_len, check_h, check_w, wan_num_frames_cfg, self.deblur_prefix_frames, self.dynamics_source,
        )

    def _trainable_parameters(self):
        # MUST stay in sync with what __init__ appends to self.models. A module
        # registered in self.models but omitted here is SAVED and reported as
        # "trainable" by printize_model_memory (it checks requires_grad), and it
        # even accumulates .grad -- but it is never handed to AdamW, so it is
        # never stepped and its weights stay at init forever. For the two
        # zero-init modules below that means an exact, permanent no-op wearing
        # a perfectly healthy loss curve.
        mods = [self.dynamics_connector, self.modulation, self.pga]
        if self.dynamics_xattn is not None:
            mods.append(self.dynamics_xattn)
        if self.blur_cond_embed is not None:
            mods.append(self.blur_cond_embed)
        if self.predictor_trainable and self.dynamics_encoder is not None:
            mods.append(self.dynamics_encoder)     # MUST be here or its LoRA is saved but never stepped
        params = []
        for m in mods:
            params.extend(p for p in m.parameters() if p.requires_grad)
        if self.lora_finetune:
            params.extend(p for p in self.transformer.parameters() if p.requires_grad)
        return params

    def after_resume_checkpoint(self, load_path):
        """Force the CONFIG's learning rate onto the resumed optimizer +
        scheduler, instead of silently inheriting the checkpoint's.

        Needed because `accelerator.load_state()` restores `initial_lr` /
        `base_lrs` from the checkpoint, and a LambdaLR computes
        `lr = initial_lr * lambda(step)` -- so resuming ckpt-140000 (which
        stores initial_lr=1e-4, base_lrs=[1e-4], and only reached lr=2e-5
        because of its cosine_floor schedule) under a fresh `constant`
        schedule would run at 1e-4, i.e. 5x the intended fine-tuning rate,
        with no warning. Verified directly against that checkpoint's
        optimizer.bin/scheduler.bin before writing this. Same class of
        silent no-op documented for the rest of this repo in
        findings/ and the accelerate-save_optimizer notes; the unwrap
        details below are the empirically-verified ones (AcceleratedOptimizer
        proxies .param_groups live, AcceleratedScheduler does NOT expose
        base_lrs/_last_lr and needs the inner .scheduler).

        Opt out by setting `train.optim.override_lr_on_resume: false`.
        """
        if not bool(_cfg(self.opt.train.optim, "override_lr_on_resume", True)):
            return
        new_lr = float(self.opt.train.optim.learning_rate)
        for opt_ in self.optimizers:
            for group in opt_.param_groups:
                g_lr = new_lr * float(group.get("lr_scale", 1.0))
                group["lr"] = g_lr
                if "initial_lr" in group:
                    group["initial_lr"] = g_lr
        for sched in self.schedulers:
            inner = getattr(sched, "scheduler", sched)  # unwrap AcceleratedScheduler
            scales = [float(g.get("lr_scale", 1.0)) for g in self.optimizers[0].param_groups] if self.optimizers else []
            if hasattr(inner, "base_lrs"):
                inner.base_lrs = [new_lr * (scales[i] if i < len(scales) else 1.0) for i in range(len(inner.base_lrs))]
            if hasattr(inner, "_last_lr"):
                inner._last_lr = [new_lr * (scales[i] if i < len(scales) else 1.0) for i in range(len(inner._last_lr))]
        self.logger.info(
            "after_resume_checkpoint: forced resumed LR to %.3g (checkpoint's own initial_lr ignored)",
            new_lr,
        )

    def setup_optimizers(self):
        opt = self.opt.train.optim
        trainable_params = self._trainable_parameters()
        # Fully-trained (modules_to_save) transformer params get a scaled LR: they start from pretrained weights, so
        # the LoRA-sized 1e-4 would be destructive. `lr_scale` is stored IN the group so after_resume_checkpoint keeps it.
        full_scale = float(_cfg(opt, "unfreeze_lr_scale", 0.1))
        full_ids = {id(p) for n, p in self.transformer.named_parameters() if "modules_to_save" in n and p.requires_grad} \
            if self.lora_finetune else set()
        if full_ids:
            groups = [
                {"params": [p for p in trainable_params if id(p) not in full_ids], "lr_scale": 1.0, "lr": opt.learning_rate},
                {"params": [p for p in trainable_params if id(p) in full_ids], "lr_scale": full_scale,
                 "lr": opt.learning_rate * full_scale},
            ]
            self.logger.info("optimizer: %d LoRA/adapter tensors @ lr, %d fully-trained transformer tensors @ lr*%.3g",
                             len(groups[0]["params"]), len(groups[1]["params"]), full_scale)
            trainable_params = groups
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=_cfg(opt, "adam_weight_decay", 0.01),
            eps=_cfg(opt, "adam_epsilon", 1e-8),
        )
        self.optimizers.append(optimizer)

    def _sample_burst_size(self, is_train: bool) -> int:
        """The burst size for THIS step.

        Curriculum: the upper bound grows min -> max linearly over
        `ramp_steps` (0 = no ramp, full range immediately), and within the
        currently-allowed range P(S) is proportional to S**-weight_power, so
        smaller bursts dominate early and stay over-represented throughout.

        Seeded on global_step so every DDP rank picks the SAME S without a
        collective -- ranks disagreeing would give mismatched tensor shapes
        and hang the all-reduce.

        Validation always uses a fixed size, otherwise the val curve mixes
        difficulties and is not comparable across checkpoints.
        """
        d = self.dyn_burst
        if not is_train:
            return int(d["val_size"] or d["max"])
        lo, hi = d["min"], d["max"]
        if d["ramp_steps"] > 0:
            elapsed = max(0, int(self.global_step) - d["ramp_start_step"])
            frac = min(1.0, elapsed / float(d["ramp_steps"]))
            hi = lo + int(round(frac * (hi - lo)))
        sizes = np.arange(lo, hi + 1)
        w = sizes.astype(np.float64) ** (-d["weight_power"])
        w /= w.sum()
        rng = np.random.default_rng(d["seed"] + int(self.global_step))
        return int(rng.choice(sizes, p=w))

    def _truncate_burst(self, data, S: int):
        """Take the first S segments of a max-burst sample, plus the matching
        GT-video prefix. Returns a NEW dict; the input is left alone."""
        d = self.dyn_burst
        out = dict(data)
        s_max = data["burst_blur"].shape[1]
        if S > s_max:
            raise ValueError(f"sampled burst size {S} > dataset burst size {s_max}")
        if S == s_max:
            return out
        f_max = data["wan_frames"].shape[1]
        f_s = f_max - d["center_stride"] * (s_max - S)
        if f_s < 1:
            raise ValueError(f"burst size {S} gives non-positive frame count {f_s}")
        out["burst_blur"] = data["burst_blur"][:, :S]
        if "burst_frames" in data:
            out["burst_frames"] = data["burst_frames"][:, :S]
        out["wan_frames"] = data["wan_frames"][:, :f_s]
        return out

    def feed_data(self, data, is_train=True):
        if self.dyn_burst is not None:
            S = self._sample_burst_size(is_train)
            data = self._truncate_burst(data, S)
            self.current_burst_size = S
        else:
            self.current_burst_size = self.burst_size
        self.sample = data
        self._check_pm1(data, is_train)
        patched = self.opt.train.patched if is_train else self.opt.val.patched
        if patched:
            raise NotImplementedError("Patched/grid training is not supported for Wan22VJEPADynamics_model.")

    def _check_pm1(self, data, is_train):
        """The model assumes wan_frames / context frames in [-1,1]. A [0,1] tensor (e.g. a manifest with wan_frames_range '01') is washed by
        (x+1)/2 with NO error. A real pm1 clip of the batch has dark pixels (min < 0), a [0,1] tensor has min >= 0 and max <= 1.
        Eval raises (an invalid eval is worse than a crash); training only warns once (a very bright crop could trip it legitimately)."""
        if not isinstance(data, dict):
            return
        for key in ("wan_frames", getattr(self, "context_key", None)):
            x = data.get(key) if key else None
            if x is None or not torch.is_tensor(x):
                continue
            lo, hi = float(x.min()), float(x.max())
            if lo >= -1e-4 and hi <= 1.0 + 1e-4:
                msg = (f"{key} range [{lo:.3f},{hi:.3f}] looks like [0,1] data but the model expects [-1,1] "
                       f"(wan_frames_range '01' manifest?) -- the GT / V-JEPA oracle input would be washed")
                if not is_train:
                    raise ValueError(msg)
                if not getattr(self, "_warned_pm1", False):
                    self._warned_pm1 = True
                    self.logger.warning(msg)

    # --- dynamics tokens (source depends on self.dynamics_source) ---

    def _vj_extract(self, clip):
        """Frozen V-JEPA tokens for the ORACLE (vjepa_gt) paths: [B, N, D] or, with vjepa_layers (K>1), [B, N, D*K]."""
        if len(self.vjepa_layers) > 1:
            return extract_vjepa_tokens_layers(self.vjepa, clip, self.accelerator.device, self.vjepa_dtype, self.vjepa_layers)
        return extract_vjepa_tokens(self.vjepa, clip, self.accelerator.device, self.vjepa_dtype)

    def _dynamics_full_grid_burst(self, sample):
        """Burst-mode analogue of _dynamics_full_grid: runs the exact same
        per-segment logic burst_size times (once per element of the
        temporally-ordered burst) and concatenates the resulting token
        grids along the TUBELET axis (dim=1) -- i.e. treats the burst as one
        long virtual clip with burst_size*per_segment_vjepa_num_tubelets
        total tubelets. self.vjepa_num_tubelets is already set to that TOTAL
        (see __init__), so every function downstream of this one
        (pool_dynamics_to_frame_tokens / resample_dynamics_grid / the
        connector / broadcast) sees one big grid and needs no burst-specific
        code at all.

        IMPORTANT (advisor-reviewed): for dynamics_source="vjepa_gt", V-JEPA
        runs on each segment's OWN real vjepa_window_size sub-window
        independently -- NOT once on the whole concatenated burst video.
        This is required so Stage 1's oracle produces a grid with the same
        layout/distribution Stage 3's per-segment predictor will produce at
        deployment; a whole-clip oracle pass would use a different tubelet
        count and content that Stage 3 could never reproduce (see
        findings/: "per-segment-then-concatenated, not whole-clip")."""
        if self.disable_dynamics_injection and self.dynamics_source == "vjepa_gt":
            # P2a no-dynamics baseline: the grid is zeroed downstream anyway, so do not pay for the V-JEPA forward.
            _b, _s = sample["burst_blur"].shape[0], sample["burst_blur"].shape[1]
            _n = _s * int(self.per_segment_vjepa_num_tubelets) * self.vjepa_grid_h * self.vjepa_grid_w
            return torch.zeros(_b, _n, self.vjepa_feature_dim * self.vjepa_num_layers, device=self.accelerator.device, dtype=torch.float32), None
        if self.dynamics_source == "blur_jepa":
            burst_blur = sample["burst_blur"].to(self.accelerator.device, dtype=self.weight_dtype)  # [B,S,3,H,W]
            # S from the TENSOR, not self.burst_size: under dynamic_burst
            # feed_data truncates the sample, so self.burst_size is only the
            # maximum. Reading the shape keeps both paths correct either way.
            s_count = burst_blur.shape[1]

            self._set_stage_c_cue(sample)
            if (self.dynamics_predictor_type == "vjepa_lora"
                    and getattr(self, "dynamics_burst_vjepa_input", "per_segment") == "single_clip"):
                # ONE ordered clip, ONE forward -- the exact construction
                # BlurJepaDynamics_model._predict_tokens_burst_single_clip uses,
                # so a predictor trained with burst_vjepa_input: single_clip
                # sees at deployment what it saw at training.
                with torch.set_grad_enabled(self.predictor_trainable and torch.is_grad_enabled()):
                    segs, viz_segs = [], []
                    for s_i in range(s_count):
                        blur01 = (burst_blur[:, s_i].float() + 1.0) / 2.0
                        pre = vjepa_from_unit_tensor(
                            blur01, size=self.vjepa_image_size, device=self.accelerator.device,
                            out_dtype=self.vjepa_dtype,
                        )  # [B,3,H,W]
                        segs.append(pre.unsqueeze(2).repeat(
                            1, 1, self.dynamics_vjepa_repeat_frames, 1, 1))
                        # Visualization only: the SAME repeated content pre-letterbox/pre-ImageNet-norm,
                        # at the blur's own resolution (not re-resized to vjepa_image_size) -- for
                        # log_video's stacked panel, not for the encoder.
                        viz_segs.append(blur01.unsqueeze(2).repeat(1, 1, self.dynamics_vjepa_repeat_frames, 1, 1))
                    clip = torch.cat(segs, dim=2)  # [B,3,S*repeat,H,W] -- ORDERED
                    viz_clip01 = torch.cat(viz_segs, dim=2).permute(0, 2, 1, 3, 4)  # [B,S*repeat,3,H,W] in [0,1]
                    return extract_vjepa_tokens(
                        self.dynamics_encoder, clip, self.accelerator.device, self.vjepa_dtype
                    ), viz_clip01

            grids, viz_segs = [], []
            with torch.no_grad():
                for s in range(s_count):
                    blur_image = burst_blur[:, s]
                    if self.dynamics_predictor_type == "vjepa_lora":
                        blur01 = (blur_image.float() + 1.0) / 2.0
                        preprocessed = vjepa_from_unit_tensor(
                            blur01, size=self.vjepa_image_size, device=self.accelerator.device,
                            out_dtype=self.vjepa_dtype,
                        )
                        clip = preprocessed.unsqueeze(2).repeat(1, 1, self.dynamics_vjepa_repeat_frames, 1, 1)
                        grid_s = extract_vjepa_tokens(
                            self.dynamics_encoder, clip, self.accelerator.device, self.vjepa_dtype
                        )
                        viz_segs.append(blur01.unsqueeze(2).repeat(1, 1, self.dynamics_vjepa_repeat_frames, 1, 1))
                    else:
                        grid_s = self.dynamics_encoder(blur_image)
                    grids.append(grid_s)
            viz_clip01 = torch.cat(viz_segs, dim=2).permute(0, 2, 1, 3, 4) if viz_segs else None
            return torch.cat(grids, dim=1), viz_clip01

        return self._oracle_grid_burst(sample)

    @torch.no_grad()
    def _oracle_tokens_s1(self, sample, reverse: bool = False):
        """Frozen-V-JEPA tokens [B,N,D] of the GT clip (even prefix of wan_frames), forward or time-reversed. S=1 only."""
        dev = self.accelerator.device
        c01 = (sample["wan_frames"].to(dev).float() + 1.0) / 2.0
        c01 = c01[:, : (c01.shape[1] // 2) * 2]
        if reverse:
            c01 = c01.flip(1)
        b, t = c01.shape[:2]
        pre = vjepa_from_unit_tensor(c01.reshape(b * t, *c01.shape[2:]), size=self.vjepa_image_size, device=dev, out_dtype=self.vjepa_dtype)
        clip = pre.reshape(b, t, 3, self.vjepa_image_size, self.vjepa_image_size).permute(0, 2, 1, 3, 4)
        return self._vj_extract(clip)

    def _apply_blend(self, grid, sample, idx):
        """Token-quality / direction response arms (S=1, dynamics_source=blur_jepa, load_oracle_vjepa: true). `grid` = Stage-B tokens B.
          token_blend  (alpha, target fwd|rev): (1-alpha) B + alpha O, O = oracle tokens of the forward / time-reversed GT clip
          axis_oracle : B with its coordinate on the F<->R axis set to the FORWARD oracle's (rest of B untouched)
          oracle_axis_B: forward oracle F with its F<->R axis coordinate replaced by B's (rest of F untouched)
        axis coordinate t(X) = <X - (F+R)/2, F-R> / |F-R|^2  (F: +0.5, R: -0.5), computed per clip over the whole token set."""
        arm = self.active_arm
        mode = arm["mode"] if (self.suite is not None and arm) else None
        if mode not in ("token_blend", "axis_oracle", "oracle_axis_B"):
            return grid
        if self.burst_size != 1 or self.dynamics_source != "blur_jepa" or self.vjepa is None:
            raise RuntimeError("blend arms need S=1, dynamics_source=blur_jepa and load_oracle_vjepa: true")
        Bt = grid.float()
        if mode == "token_blend":
            O = self._oracle_tokens_s1(sample, reverse=(str(arm.get("target") or "fwd") == "rev")).float()
            a = float(arm["alpha"])
            out = (1 - a) * Bt + a * O
        else:
            F_, R_ = self._oracle_tokens_s1(sample, False).float(), self._oracle_tokens_s1(sample, True).float()
            ax, mid = F_ - R_, 0.5 * (F_ + R_)
            den = (ax * ax).sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            tB = ((Bt - mid) * ax).sum(dim=(1, 2), keepdim=True) / den
            out = Bt + (0.5 - tB) * ax if mode == "axis_oracle" else F_ + (tB - 0.5) * ax
        os.makedirs(self.opt.path.experiments_root, exist_ok=True)
        import json as _j
        with open(os.path.join(self.opt.path.experiments_root, "blend_log.jsonl"), "a") as f:
            f.write(_j.dumps({"idx": int(idx), "arm": arm["name"], "mode": mode, "alpha": arm.get("alpha"), "target": arm.get("target"),
                              "change_rel": float((out - Bt).norm() / Bt.norm())}) + "\n")
        return out.to(grid.dtype)

    def _edit_mode(self) -> str:
        """Arm mode (suite) or config default: none | edit_keep | edit_gt_fwd | edit_gt_rev | edit_gt_region_rev | edit_json."""
        am = self.active_arm["mode"] if self.active_arm else None
        if am and am.startswith("edit_"):
            return am
        cfg = getattr(self, "dynamics_edit_control", "none")
        return {"none": "none", "gt_fwd": "edit_gt_fwd", "gt_rev": "edit_gt_rev"}.get(cfg, "edit_json" if cfg.startswith("json:") else "none")

    def _apply_editor(self, grid, sample, idx):
        """Edit the PREDICTED tokens (Stage B output) with the user control map. S=1, dynamics_source=blur_jepa only.
        Control sources: edit_keep (empty control -> must be an exact no-op), edit_gt_fwd / edit_gt_rev (GT-flow regions asking for
        the GT direction / its opposite), edit_gt_region_rev (arm.seg = region index reversed, the others asked forward),
        edit_json (regions from a control JSON: arm.control or dynamics_edit_control: json:<path>)."""
        mode = self._edit_mode()
        if mode == "none":
            return grid
        ed = getattr(self, "dynamics_editor", None)
        if ed is None:
            raise RuntimeError(f"edit mode {mode!r} needs dynamics_editor_checkpoint")
        if self.burst_size != 1 or self.dynamics_source != "blur_jepa":
            raise NotImplementedError("the token editor is S=1 / dynamics_source=blur_jepa only")
        import json as _json
        from models.reFlux.motion_cue import build_control_map, mask_to_cells, sample_edit_regions
        dev = self.accelerator.device
        blur = sample["blur"].to(dev, dtype=torch.float32)                          # [B,3,H,W] in [-1,1]
        B, _, H, W = blur.shape
        ctl = np.zeros((B, 24, 24, 2), np.float32)
        n_reg = 0
        for b in range(B):
            if mode == "edit_keep":
                continue
            if mode == "edit_json":
                path = (self.active_arm or {}).get("control") if self.active_arm else None
                path = path or getattr(self, "dynamics_edit_control", "none").split("json:", 1)[-1]
                ctl[b] = build_control_map((H, W), _json.load(open(path))["regions"])
                continue
            fr = ((sample["wan_frames"][b, :8].float().cpu().numpy() + 1.0) / 2.0).clip(0, 1).transpose(0, 2, 3, 1)   # [8,H,W,3]
            regs = sample_edit_regions(fr, np.random.default_rng(1000 + int(idx) + b), p_user_style=0.0)
            regs = [r for r in regs if r["d_fwd"] is not None]
            n_reg = len(regs)
            k_rev = int((self.active_arm or {}).get("seg") or 0) if mode == "edit_gt_region_rev" else None
            for ri, r in enumerate(regs):
                sign = -1.0 if (mode == "edit_gt_rev" or (mode == "edit_gt_region_rev" and ri == k_rev)) else 1.0
                ctl[b][mask_to_cells(r["mask"])] = sign * r["d_fwd"]
        control = torch.from_numpy(ctl).to(dev)
        canvas = vjepa_from_unit_tensor((blur + 1.0) / 2.0, size=self.vjepa_image_size, device=dev, out_dtype=torch.float32)
        with torch.no_grad():
            out = ed(grid.float(), canvas, control)
        os.makedirs(self.opt.path.experiments_root, exist_ok=True)
        with open(os.path.join(self.opt.path.experiments_root, "edit_log.jsonl"), "a") as f:
            f.write(_json.dumps({"idx": int(idx), "arm": self.active_arm["name"] if self.active_arm else None, "mode": mode, "n_regions": n_reg,
                                 "controlled_cells": int((np.abs(ctl).sum(-1) > 0).sum()), "max_abs_change": float((out - grid.float()).abs().max())}) + "\n")
        return out.to(grid.dtype)

    def _stage_c_cue_mode(self) -> str:
        """Config default, overridden per suite arm: modes cue_gt / cue_gt_flip run the predictor with the GT-flow cue (+d / -d)."""
        am = self.active_arm["mode"] if self.active_arm else None
        return {"cue_gt": "gt_flow", "cue_gt_flip": "gt_flow_flip"}.get(am, getattr(self, "dynamics_cue_mode", "none"))

    def _set_stage_c_cue(self, sample):
        cue = getattr(self, "dynamics_cue", None)
        mode = self._stage_c_cue_mode()
        if cue is None or mode == "none":
            if cue is not None:
                cue.set_cue(None)
            return
        from models.reFlux.motion_cue import cue_from_burst
        m, d = cue_from_burst(sample["burst_frames"].to(self.accelerator.device))
        cue.set_cue(m, -d if mode == "gt_flow_flip" else d)

    def _oracle_grid_burst(self, sample):
        """ORACLE tokens: frozen V-JEPA (or DINOv2, P2c) on each segment's REAL frames -> (grid, full clip)."""
        # vjepa_gt: each segment's own real vjepa_window_size-length
        # sub-window (Adobe240Dataset's burst_frames_key), concatenated.
        burst_frames = sample["burst_frames"].to(self.accelerator.device)  # [B,S,T,3,H,W] in [0,1]
        bsz, s_count = burst_frames.shape[0], burst_frames.shape[1]
        # INPUT-level temporal interventions (eval suite only): permute the REAL frames BEFORE the frozen
        # V-JEPA sees them, unlike the token-level arms in _apply_token_arm which permute V-JEPA's output.
        # The blur / first-frame condition is untouched (it is read from burst_blur, not burst_frames).
        _am = self.active_arm["mode"] if self.active_arm else None
        if _am in ("input_reverse", "input_reverse_blur", "input_reverse_within_seg", "input_shuffle", "input_shuffle_global"):
            _T = burst_frames.shape[2]
            _gen = torch.Generator(device="cpu").manual_seed(int(self.sampler_seed or 0) * 104729 + 7)
            if _am in ("input_reverse", "input_reverse_blur"):            # whole burst played backwards: segment order AND frames within
                burst_frames = burst_frames.flip(1).flip(2)
            elif _am == "input_reverse_within_seg":
                burst_frames = burst_frames.flip(2)
            elif _am == "input_shuffle":          # random frame order inside each segment
                burst_frames = torch.stack(
                    [burst_frames[:, s_i][:, torch.randperm(_T, generator=_gen).to(burst_frames.device)]
                     for s_i in range(s_count)], dim=1)
            else:                                 # input_shuffle_global: shuffle all S*T frames, then re-cut
                _flat = burst_frames.reshape(bsz, s_count * _T, *burst_frames.shape[3:])
                _flat = _flat[:, torch.randperm(s_count * _T, generator=_gen).to(_flat.device)]
                burst_frames = _flat.reshape(burst_frames.shape)
        grids = []
        clips_for_log = []
        for s in range(s_count):
            clip01 = burst_frames[:, s].float()  # [B,T,3,H,W] in [0,1]
            t_seg = clip01.shape[1]
            flat = clip01.reshape(bsz * t_seg, *clip01.shape[2:])
            preprocessed = vjepa_from_unit_tensor(
                flat, size=self.vjepa_image_size, device=self.accelerator.device, out_dtype=self.vjepa_dtype,
            )
            clip = preprocessed.reshape(bsz, t_seg, 3, self.vjepa_image_size, self.vjepa_image_size)
            clip = clip.permute(0, 2, 1, 3, 4)  # [B,3,T,H,W]
            with torch.no_grad():
                if self.dynamics_encoder_type == "dinov2":
                    grid_s = self._dino_tokens(clip01)
                else:
                    grid_s = self._vj_extract(clip)
            grids.append(grid_s)
            clips_for_log.append(clip01)
        grid = torch.cat(grids, dim=1)
        if _am in ("input_reverse", "input_reverse_blur", "input_reverse_within_seg", "input_shuffle", "input_shuffle_global"):
            self._log_token_probe(sample, grid)
        # Concatenated along time for visualization only (segments overlap
        # in reality -- this approximates the full clip, it is not re-derived
        # from the true continuous span, which validate_step never needs).
        full_clip01 = torch.cat(clips_for_log, dim=1)  # [B, S*T, 3, H, W]
        return grid, full_clip01

    @torch.no_grad()
    def _log_token_probe(self, sample, grid_perm):
        """Direct V-JEPA probe (no Wan): are the tokens of the permuted video the time-flip of the original's
        tokens (token_cos_flip high, cos_same low) -> V-JEPA encodes direction; or merely similar to the
        original's tokens (cos_same >= cos_flip) -> V-JEPA tokens are ~time-symmetric. Per clip, per segment
        (tubelet axis flipped within a segment; for input_reverse the segment order is flipped too)."""
        import json
        arm = self.active_arm
        self.active_arm = None
        try:
            grid_orig, _ = self._oracle_grid_burst(sample) if self.burst_size > 1 else self._dynamics_full_grid(sample)
        finally:
            self.active_arm = arm
        B, N, D = grid_orig.shape
        cells = self.vjepa_grid_h * self.vjepa_grid_w
        T = N // cells
        P = int(self.per_segment_vjepa_num_tubelets)
        S = max(T // P, 1)
        o = grid_orig.float().reshape(B, S, P, cells, D)
        q = grid_perm.float().reshape(B, S, P, cells, D)
        cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        rec = {"arm": arm["name"], "idx": len(self._clip_stems),
               "cos_same": cos(q, o),
               "cos_flip_within_seg": cos(q, o.flip(2)),
               "cos_flip_all": cos(q, o.flip(1).flip(2)),
               "cos_orig_vs_orig_flip_within_seg": cos(o, o.flip(2)),   # self-similarity floor of the flip itself
               "cos_orig_vs_orig_flip_all": cos(o, o.flip(1).flip(2))}
        os.makedirs(self.opt.path.experiments_root, exist_ok=True)
        with open(os.path.join(self.opt.path.experiments_root, "probe.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _dynamics_full_grid(self, sample):
        """-> (grid [B, vjepa_num_tubelets*vjepa_grid_h*vjepa_grid_w, vjepa_feature_dim] no grad,
        vjepa_input_clip_01 [B,T,3,H,W] in [0,1] or None). The second element is the raw
        (pre-ImageNet-normalization) clip actually handed to V-JEPA -- only meaningful for
        dynamics_source="vjepa_gt" (None for "blur_jepa", which never runs V-JEPA on video) --
        exposed purely so callers (validate_step) can log exactly what V-JEPA saw."""
        if self.burst_size > 1:
            return self._dynamics_full_grid_burst(sample)
        if self.dynamics_source == "blur_jepa":
            blur_image = sample["blur"].to(self.accelerator.device, dtype=self.weight_dtype)
            self._set_s1_control(sample)
            try:
                return self._dynamics_full_grid_s1_blur(blur_image)
            finally:
                self._clear_s1_control()
        return self._dynamics_full_grid_s1_oracle(sample)

    def _stroke_mode(self) -> str:
        am = self.active_arm["mode"] if self.active_arm else None
        return {"stroke_fwd": "gt", "stroke_rev": "gt_rev", "stroke_shuffle": "gt_shuffle", "stroke_none": "none"}.get(
            am, getattr(self, "dynamics_stroke_mode", "none"))

    @torch.no_grad()
    def _set_s1_control(self, sample):
        """S=1 Stage C: arm the predictor's optional control from the GT clip (wan_frames[:, :8], the frames the S=1
        oracle sees). Old global cue (cue_gt / cue_gt_flip) or user strokes (stroke_fwd / stroke_rev / stroke_shuffle)."""
        dev = self.accelerator.device
        cue = getattr(self, "dynamics_cue", None)
        cue_mode = self._stage_c_cue_mode()
        if cue is not None and cue_mode != "none":
            from models.reFlux.motion_cue import cue_from_burst
            g01 = ((sample["wan_frames"][:, :8].to(dev).float() + 1.0) / 2.0).clamp(0, 1)
            m, d = cue_from_burst(g01.unsqueeze(1))
            cue.set_cue(m, -d if cue_mode == "gt_flow_flip" else d)
        ctrl = getattr(self, "dynamics_stroke", None)
        mode = self._stroke_mode()
        if ctrl is None or mode == "none":
            return
        from models.reFlux import stroke_control as _sc
        import json as _json
        idx0 = int(getattr(self, "_cur_val_idx", 0))
        g01 = ((sample["wan_frames"][:, :8].to(dev).float() + 1.0) / 2.0).clamp(0, 1)
        B = g01.shape[0]
        rngs = [np.random.default_rng([int(_cfg(self.opt, "dynamics_stroke_seed", 0)), idx0 + i]) for i in range(B)]
        strokes = _sc.extract_strokes(self.dynamics_stroke_raft, g01, rngs, self.dynamics_stroke_cfg)
        fwd = strokes
        if mode == "gt_rev":
            strokes = [_sc.reverse_strokes(s) for s in strokes]
        elif mode == "gt_shuffle":
            prev = self._prev_wan_strokes
            strokes = [prev if prev else [] for _ in range(B)]
        if fwd and fwd[-1]:
            self._prev_wan_strokes = fwd[-1]
        H, W = sample["blur"].shape[-2:]
        has = torch.tensor([len(s) > 0 for s in strokes], device=dev)
        if bool(has.any()):
            ctrl.set_control(_sc.render_strokes(strokes, (H, W), self.vjepa_image_size, dev,
                                                band=getattr(self, "dynamics_stroke_band", None)), has)
        os.makedirs(self.opt.path.experiments_root, exist_ok=True)
        with open(os.path.join(self.opt.path.experiments_root, "stroke_log.jsonl"), "a") as f:
            for i in range(B):
                f.write(_json.dumps({"idx": idx0 + i, "mode": mode, "n_strokes": len(strokes[i]),
                                     "strokes": [{"pts": s["pts"].tolist(), "thick": s["thick"]} for s in strokes[i]]}) + "\n")
        try:   # what the predictor was told, for eyeballing
            import cv2
            for i in range(B):
                bl = ((sample["blur"][i].float() + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                img = _sc.draw_strokes(bl, strokes[i])
                cv2.imwrite(os.path.join(self.opt.path.experiments_root, f"strokes_{idx0 + i:04d}.jpg"),
                            (img[..., ::-1] * 255).astype(np.uint8))
        except Exception as e:
            self.logger.warning("stroke image failed: %s", e)

    def _clear_s1_control(self):
        if getattr(self, "dynamics_cue", None) is not None:
            self.dynamics_cue.set_cue(None)
        if getattr(self, "dynamics_stroke", None) is not None:
            self.dynamics_stroke.set_control(None)

    def _dynamics_full_grid_s1_blur(self, blur_image):
        with torch.no_grad():
            if self.dynamics_predictor_type == "vjepa_lora":
                # Same repeated-frame "fake clip" feeding convention
                # BlurJepaDynamics_model._predict_tokens uses for this
                # predictor type: the blur image, in [0,1], resized/
                # normalized once, then tiled across
                # dynamics_vjepa_repeat_frames IDENTICAL copies so
                # V-JEPA's own tubelet_size=2 grouping produces the same
                # token-grid shape this checkpoint was trained against.
                blur01 = (blur_image.float() + 1.0) / 2.0
                preprocessed = vjepa_from_unit_tensor(
                    blur01, size=self.vjepa_image_size, device=self.accelerator.device,
                    out_dtype=self.vjepa_dtype,
                )  # [B,3,H,W]
                clip = preprocessed.unsqueeze(2).repeat(1, 1, self.dynamics_vjepa_repeat_frames, 1, 1)
                # Visualization only (see the burst-mode branches above for the same pattern).
                viz_clip01 = blur01.unsqueeze(1).repeat(1, self.dynamics_vjepa_repeat_frames, 1, 1, 1)
                return extract_vjepa_tokens(
                    self.dynamics_encoder, clip, self.accelerator.device, self.vjepa_dtype
                ), viz_clip01
            return self.dynamics_encoder(blur_image), None

    def _dynamics_full_grid_s1_oracle(self, sample):
        # vjepa_gt: V-JEPA sees exactly the clip Wan is being asked to
        # generate -- an oracle signal for architecture validation only.
        wan_frames_pm1 = sample["wan_frames"].to(self.accelerator.device)
        wan_frames_01 = (wan_frames_pm1.float() + 1.0) / 2.0  # [B,T,3,H,W] in [0,1]
        t_full = wan_frames_01.shape[1]
        even_t = (t_full // 2) * 2  # V-JEPA's tubelet_size=2 needs an even frame count
        clip01 = wan_frames_01[:, :even_t]
        # INPUT-space reversal (suite arm, S=1 oracle): play the clip V-JEPA sees backwards. Mirrors the burst-path
        # input_reverse in _oracle_grid_burst; without this the arm would silently be a no-op at S=1.
        _am1 = self.active_arm["mode"] if self.active_arm else None
        if _am1 in ("input_reverse", "input_reverse_within_seg"):
            clip01 = clip01.flip(1)
        b = clip01.shape[0]
        flat = clip01.reshape(b * even_t, *clip01.shape[2:])
        preprocessed = vjepa_from_unit_tensor(
            flat, size=self.vjepa_image_size, device=self.accelerator.device, out_dtype=self.vjepa_dtype,
        )
        clip = preprocessed.reshape(b, even_t, 3, self.vjepa_image_size, self.vjepa_image_size)
        clip = clip.permute(0, 2, 1, 3, 4)  # [B,3,T,H,W]
        with torch.no_grad():
            grid = self._vj_extract(clip)
        if _am1 in ("input_reverse", "input_reverse_within_seg"):
            self._log_token_probe(sample, grid)
        return grid, clip01

    def _build_dynamics_tokens(self, dynamics_grid, post_patch_num_frames, post_patch_height, post_patch_width) -> torch.Tensor:
        if self.dynamics_connector_type == "pooled":
            # Original pipeline: global-average-pools V-JEPA's spatial grid
            # away, one vector per frame broadcast uniformly across that
            # frame's spatial tokens -- required to correctly evaluate any
            # checkpoint trained before "factored" existed.
            per_frame = pool_dynamics_to_frame_tokens(
                dynamics_grid, self.vjepa_num_tubelets, self.vjepa_grid_h, self.vjepa_grid_w, post_patch_num_frames,
            )
            projected = self.dynamics_connector(per_frame)  # trainable, [B,T,wan_inner_dim]
            return broadcast_frame_tokens_to_video_tokens(projected, post_patch_height * post_patch_width)

        # "factored": preserves V-JEPA's spatial grid (bilinearly resized to
        # Wan's own post-patch H,W, upsampling or downsampling as needed --
        # a no-op when the sizes already match) instead of pooling it away,
        # so different spatial positions inject different content.
        grid_thw = resample_dynamics_grid(
            dynamics_grid, self.vjepa_num_tubelets, self.vjepa_grid_h, self.vjepa_grid_w,
            post_patch_num_frames, post_patch_height, post_patch_width,
        )  # [B, T, H, W, vjepa_feature_dim]
        projected = self.dynamics_connector(grid_thw)  # trainable, [B, T, H, W, wan_inner_dim]
        b = projected.shape[0]
        return projected.reshape(b, post_patch_num_frames * post_patch_height * post_patch_width, self.wan_inner_dim)

    def _first_frame_condition(self, sample) -> torch.Tensor:
        """[B,3,H,W] in [-1,1]: the actual accessible I2V conditioning
        image. Single-frame mode: the one blurred input. Burst mode: the
        FIRST burst segment's blurred input -- the deblur-priming prefix
        (deblur_prefix_frames) is applied only ONCE, at the very start of
        the whole burst, exactly mirroring the single-frame mechanism
        rather than repeating it per segment."""
        if self.burst_size > 1:
            return sample["burst_blur"][:, 0].to(self.accelerator.device, dtype=torch.float32)
        return sample["blur"].to(self.accelerator.device, dtype=torch.float32)

    @staticmethod
    def _limit_burst_resolution(burst: torch.Tensor, size: int) -> torch.Tensor:
        """[B,S,3,H,W] -> same shape, but carrying only `size`-resolution detail:
        down to longer side == `size` (bicubic+antialias, the same resize V-JEPA's
        `vjepa_from_unit_tensor` applies) and back up to H x W, so the VAE grid and
        block-0 token layout are unchanged. Makes burst_blur_conditioning input-
        resolution-matched to the V-JEPA arm (vjepa_image_size)."""
        b, s, c, h, w = burst.shape
        if max(h, w) <= size:
            return burst
        k = size / max(h, w)
        dh, dw = max(1, round(h * k)), max(1, round(w * k))
        flat = burst.reshape(b * s, c, h, w)
        down = F.interpolate(flat, size=(dh, dw), mode="bicubic", align_corners=False, antialias=True)
        up = F.interpolate(down, size=(h, w), mode="bicubic", align_corners=False)
        return up.clamp(-1.0, 1.0).reshape(b, s, c, h, w)

    def _burst_blur_condition_video(self, sample) -> torch.Tensor:
        """[B, T, 3, H, W] in [-1,1]: the blurred input laid out in PIXEL time so
        each burst segment occupies the frames it was actually averaged from --
        n segments x r repeats, preceded by `deblur_prefix_frames` repeats of the
        FIRST segment (the prefix region's job is to deblur, and segment 1 is the
        blur that corresponds to it).

        Total length equals the target's (`_prepend_deblur_frames` output), so the
        conditioning and the denoising target are frame-aligned. Built in pixel
        space and encoded once, rather than placing latents by hand, so the
        causal VAE decides the pixel->latent mapping itself:

            p=5, n=3, r=8 -> 29 pixel frames -> 8 latents
            latents 0-3 <- blur1 (incl. prefix), 4-5 <- blur2, 6-7 <- blur3
        """
        burst = sample["burst_blur"].to(self.accelerator.device, dtype=torch.float32)  # [B,S,3,H,W]
        if self.burst_blur_cond_size:
            burst = self._limit_burst_resolution(burst, self.burst_blur_cond_size)
        s_count = burst.shape[1]
        real_frames = sample["wan_frames"].shape[1]  # authoritative: what the target actually has
        if getattr(self, "_out_frames_override", None) is not None:   # resample_time arm: generate a different length
            real_frames = int(self._out_frames_override)
        # Nearest-segment assignment for ANY frame count: frame t -> segment floor(t*S/F). The old rule demanded
        # F % S == 0 (25 % 3 != 0) and could not be combined with dynamic burst either; this gives counts 9/8/8
        # for F=25, S=3 and reduces to equal blocks whenever F is divisible by S.
        _seg = (torch.arange(real_frames, device=burst.device) * s_count) // real_frames
        video = burst[:, _seg]  # [B, F, 3, H, W]
        if self.deblur_prefix_frames > 0:
            prefix = burst[:, 0:1].expand(-1, self.deblur_prefix_frames, -1, -1, -1)
            video = torch.cat([prefix, video], dim=1)
        return video

    def _burst_blur_condition_tokens(self, sample) -> torch.Tensor | None:
        """-> [B, N, inner_dim] to add at block 0, or None when disabled."""
        if not self.burst_blur_conditioning:
            return None
        video_btchw = self._burst_blur_condition_video(sample)
        video_bcthw = video_btchw.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            blur_latent = self._encode_video_sample(video_bcthw).to(torch.float32)
        return self.blur_cond_embed(blur_latent)

    def _deblur_prefix_post_patch_frames(self) -> int:
        """How many of the sequence's post-patch temporal positions belong
        to the prepended deblur-priming region (0 if deblur_prefix_frames
        is 0)."""
        if self.deblur_prefix_frames == 0:
            return 0
        if self._prefix_latent_aligned:
            # p = 4k+1 pixel frames -> latent 0 (1 pixel) + k full latents.
            extra_t_lat = (self.deblur_prefix_frames - 1) // self.vae_scale_factor_temporal + 1
        else:
            # Legacy p % 4 == 0 path: keeps the historical (under)count so
            # existing checkpoints reproduce exactly. Note this genuinely
            # under-reports -- with p=4 the prefix also occupies 3/4 of latent
            # 1, which therefore still receives real motion tokens.
            extra_t_lat = self.deblur_prefix_frames // self.vae_scale_factor_temporal
        return extra_t_lat // self.patch_size[0]

    def _build_target_video(self, sample, wan_frames):
        """[B,T,3,H,W] exposure frames -> the full generated-video target [B,front+T+back,3,H,W] (v3 real context), or the legacy
        copy-prefix video when context_key is unset."""
        if not self.context_key:
            return self._prepend_deblur_frames(wan_frames)
        ctx = sample[self.context_key].to(self.accelerator.device, dtype=torch.float32)       # [B,Tc,3,H,W]
        T = wan_frames.shape[1]
        s0 = (ctx.shape[1] - T) // 2
        if s0 < self.ctx_front or ctx.shape[1] - (s0 + T) < self.ctx_back:
            raise ValueError(f"context window {ctx.shape[1]} too short for front={self.ctx_front}, T={T}, back={self.ctx_back}")
        return torch.cat([ctx[:, s0 - self.ctx_front:s0], wan_frames, ctx[:, s0 + T:s0 + T + self.ctx_back]], dim=1)

    def _latent_loss_weights(self, t_lat: int, T: int, device):
        """[1,1,t_lat,1,1] flow-matching weights: latents made ONLY of context frames get ctx_loss_weight, others 1. Latent 0 covers
        pixel 0; latent k>=1 covers pixels 4k-3..4k."""
        w = torch.ones(1, 1, t_lat, 1, 1, device=device)
        if not self.context_key or self.ctx_loss_weight == 1.0:
            return w
        r, e0, e1 = self.vae_scale_factor_temporal, self.ctx_front, self.ctx_front + T   # exposure = pixels [e0, e1)
        for k in range(1, t_lat):
            a, b = r * (k - 1) + 1, r * k
            if b < e0 or a >= e1:
                w[:, :, k] = self.ctx_loss_weight
        return w

    def _crop_exposure(self, video, T: int):
        """Decoded [B,3,Ttot,H,W] -> the T exposure frames."""
        return video[:, :, self.ctx_front:self.ctx_front + T]

    def _prepend_deblur_frames(self, video_btchw: torch.Tensor) -> torch.Tensor:
        """[B,T,3,H,W] -> [B,T+deblur_prefix_frames,3,H,W], prepending
        `deblur_prefix_frames` copies of the clip's own first (real, sharp)
        frame. No-op when deblur_prefix_frames == 0."""
        if self.deblur_prefix_frames == 0:
            return video_btchw
        first = video_btchw[:, 0:1]
        prefix = first.expand(-1, self.deblur_prefix_frames, -1, -1, -1)
        return torch.cat([prefix, video_btchw], dim=1)

    def _build_dynamics_kv(self, dynamics_grid, post_patch_num_frames, post_patch_height, post_patch_width):
        """Compact dynamics set for cross-attention: [B, M, inner_dim], with
        **NOTHING POOLED AND NOTHING RESAMPLED**.

        M = vjepa_num_tubelets * vjepa_grid_h * vjepa_grid_w -- the raw V-JEPA
        token grid exactly as the encoder produced it (for burst_size=3 that is
        15 x 24 x 24 = 8640 tokens, i.e. all three segments' tubelets kept
        separate and in order). The connector is applied POINTWISE (it maps the
        feature dim and broadcasts over leading dims), so it still contributes
        its trained weights without collapsing anything.

        This is the whole reason to prefer cross-attention here. The other two
        paths both destroy information before Wan sees it:
          - `pooled`   : grid.mean(dim=(2,3)) annihilates the 24x24 spatial grid,
                         leaving one vector per frame broadcast uniformly -- no
                         "where" at all.
          - both paths : then temporally interpolate vjepa_num_tubelets ->
                         post_patch_num_frames (15 -> 8 for burst-3), blending
                         adjacent tubelets and smearing the very segment
                         boundaries the burst construction exists to create.
        Cross-attention has no such constraint: the KV is a SET, so it need not
        match Wan's token count, and attention selects what each position needs.
        Ordering is carried by the module's learned kv_pos embedding.
        """
        return self.dynamics_connector(dynamics_grid)  # [B, M, inner], pointwise

    def _dynamics_tokens_with_deblur_padding(
        self, dynamics_grid, post_patch_num_frames, post_patch_height, post_patch_width,
    ) -> torch.Tensor:
        """Same as _build_dynamics_tokens, but zero-pads the *front* of the
        returned token sequence for the extra post-patch frame(s)
        contributed by the prepended deblur-priming region -- that region
        is a static repeat of the first frame with no corresponding real
        motion, so only the frames coming from the real clip get real
        V-JEPA content injected. No-op when deblur_prefix_frames == 0
        (returns _build_dynamics_tokens's output unchanged)."""
        extra_frames = self._deblur_prefix_post_patch_frames()
        real_frames = post_patch_num_frames - extra_frames
        dynamics_real = self._build_dynamics_tokens(dynamics_grid, real_frames, post_patch_height, post_patch_width)
        if extra_frames == 0:
            return dynamics_real
        tokens_per_frame = post_patch_height * post_patch_width
        zeros_prefix = torch.zeros(
            dynamics_real.shape[0], extra_frames * tokens_per_frame, dynamics_real.shape[2],
            device=dynamics_real.device, dtype=dynamics_real.dtype,
        )
        return torch.cat([zeros_prefix, dynamics_real], dim=1)

    # --- VAE / text encoding helpers ---

    def _encode_video_sample(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        video_bcthw = video_bcthw.to(self.vae.dtype)
        out = self.vae.encode(video_bcthw)
        latent = out.latent_dist.sample() if hasattr(out, "latent_dist") else out.latents
        latent = (latent - self.latents_mean) * self.latents_std
        return latent.to(memory_format=torch.contiguous_format)

    def _encode_first_frame_argmax(self, first_frame_bchw: torch.Tensor) -> torch.Tensor:
        video_condition = first_frame_bchw.unsqueeze(2).to(self.vae.dtype)  # [B,C,1,H,W]
        out = self.vae.encode(video_condition)
        latent = out.latent_dist.mode() if hasattr(out, "latent_dist") else out.latents
        latent = (latent - self.latents_mean) * self.latents_std
        return latent.to(memory_format=torch.contiguous_format)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.latents_std + self.latents_mean
        latents = latents.to(self.vae.dtype)
        return self.vae.decode(latents).sample.clamp(-1, 1)

    def _prompt_embeds(self, bsz: int) -> torch.Tensor:
        if self._empty_prompt_embeds is None:
            from models.MotionBlur2Video.Blur2Vid_Wan_model import compute_prompt_embeds_wan

            self._empty_prompt_embeds = compute_prompt_embeds_wan(
                self.tokenizer, self.text_encoder, [""], self.max_text_seq_length,
                self.accelerator.device, self.weight_dtype, requires_grad=False,
            )
            if getattr(self, "low_mem_eval", False) and not self.is_train:
                self.text_encoder = None              # the prompt is always "" -> the cached embedding is all that is ever needed
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                self.logger.info("low_mem_eval: empty-prompt embedding cached, UMT5 text encoder dropped")
        return self._empty_prompt_embeds.expand(bsz, -1, -1)

    # --- optical flow (validation-time visualization only) ---

    @torch.no_grad()
    def _flow_from_video(self, video_btchw: torch.Tensor) -> torch.Tensor:
        """video_btchw: [B,T,C,H,W] -> flow [B,T,2,H,W] (frame 0 is zero flow,
        matching Blur2Vid_Wan_with_OpticalFlow_model.calculate_optical_flow_from_video)."""
        flows = []
        for i in range(video_btchw.shape[1] - 1):
            out = self.opt_flow_model({"images": video_btchw[:, i:i + 2].to(self.opt_flow_model.dtype)})
            flows.append(out["flows"][:, 0])
        flows = torch.stack([torch.zeros_like(flows[0])] + flows, dim=1)
        return flows

    @torch.no_grad()
    def _flow_first_to_last(self, video_btchw: torch.Tensor) -> torch.Tensor:
        """video_btchw: [B,T,C,H,W] -> flow [B,2,H,W] from the first to the last frame --
        the single displacement field that best matches what actually produced the
        blur image (a time-integration of the clip over its exposure window)."""
        first_last = torch.stack([video_btchw[:, 0], video_btchw[:, -1]], dim=1)
        out = self.opt_flow_model({"images": first_last.to(self.opt_flow_model.dtype)})
        return out["flows"][:, 0]

    @staticmethod
    def _resize_flow_to(flow: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """flow: [...,2,H',W'] -> [...,2,h,w]. Rescales flow *values* to the new pixel
        scale too (not just resampling positions) -- flow vectors are in pixel units
        of whatever resolution they were computed at, matching the resize handling
        models/MotionBlur2Video/helpers.py::warp_tensor already does in this repo."""
        orig_shape = flow.shape
        h0, w0 = orig_shape[-2], orig_shape[-1]
        if (h0, w0) == (h, w):
            return flow
        flat = flow.reshape(-1, 2, h0, w0)
        resized = F.interpolate(flat.float(), size=(h, w), mode="bilinear", align_corners=False)
        resized = torch.stack([resized[:, 0] * (w / w0), resized[:, 1] * (h / h0)], dim=1)
        return resized.reshape(*orig_shape[:-2], h, w).to(flow.dtype)

    def _flow_to_video_thwc(self, flows_bt2hw: torch.Tensor) -> np.ndarray:
        """flows_bt2hw: [1,T,2,H,W] -> [T,H,W,3] float RGB visualization for logging."""
        frames = [flow_utils.flow_to_rgb(flows_bt2hw[:, i].float()) for i in range(flows_bt2hw.shape[1])]
        video = torch.cat(frames, dim=0)  # [T,3,H,W]
        return video.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    # --- V-JEPA dynamics-token PCA visualization (validation-time diagnostic only) ---

    @staticmethod
    def _pca_rgb(tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [N, D] -> [N, 3] in [0,1] -- the first 3 principal
        components (via SVD on the mean-centered tokens), min-max normalized
        per channel for display. PCA is fit on exactly the N tokens passed
        in, so the resulting RGB mapping is consistent across every position
        included in that call (e.g. every spatial position of every tubelet
        in one sample's full dynamics grid) rather than being refit
        independently per frame, which would make the color mapping flicker
        between frames of the same clip."""
        tokens = tokens.float()
        centered = tokens - tokens.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            components = vh[:3]  # [3, D]
            proj = centered @ components.T  # [N, 3]
        except Exception:
            proj = centered[:, :3]
        proj_min = proj.min(dim=0, keepdim=True).values
        proj_max = proj.max(dim=0, keepdim=True).values
        return (proj - proj_min) / (proj_max - proj_min).clamp(min=1e-6)

    def _vjepa_pca_video_thwc(self, dynamics_grid_i: torch.Tensor, n_frames: int, h: int, w: int) -> np.ndarray:
        """dynamics_grid_i: [vjepa_num_tubelets*vjepa_grid_h*vjepa_grid_w,
        vjepa_feature_dim] -- one sample's full, un-padded dynamics grid
        (real V-JEPA output for dynamics_source="vjepa_gt", predicted tokens
        for "blur_jepa"; in burst mode this is already the tubelet-axis
        concatenation across every segment -- see _dynamics_full_grid_burst,
        so a burst's PCA video shows all segments back to back exactly like
        the real burst order). Returns [n_frames, h, w, 3] float32 in
        [0,1]: a PCA-RGB visualization (see _pca_rgb) of the token grid,
        nearest-upsampled in time (vjepa_num_tubelets -> n_frames) and space
        (vjepa_grid_h x vjepa_grid_w -> h x w) to match the other videos
        logged alongside it in the validation grid."""
        gh, gw = self.vjepa_grid_h, self.vjepa_grid_w
        # Tubelet count from the TENSOR, not self.vjepa_num_tubelets. Under
        # dynamic_burst the latter is the MAXIMUM, so reshaping by it raises
        # for any S < max -- the same defect that crashed training via
        # pool_dynamics_to_frame_tokens. This only survived because validation
        # is pinned at S=max; it would fire the moment val_size changed.
        n_tub = dynamics_grid_i.shape[0] // (gh * gw)
        grid = dynamics_grid_i.reshape(n_tub, gh * gw, -1)
        rgb = self._pca_rgb(grid.reshape(-1, grid.shape[-1]))
        rgb = rgb.reshape(n_tub, gh, gw, 3)

        frame_idx = (torch.arange(n_frames, device=rgb.device) * n_tub) // max(n_frames, 1)
        frame_idx = frame_idx.clamp(max=n_tub - 1)
        rgb_t = rgb[frame_idx]  # [n_frames, gh, gw, 3]

        rgb_chw = rgb_t.permute(0, 3, 1, 2)  # [n_frames, 3, gh, gw]
        rgb_resized = F.interpolate(rgb_chw, size=(h, w), mode="nearest")
        return rgb_resized.permute(0, 2, 3, 1).cpu().float().numpy().astype(np.float32)

    @staticmethod
    def _resample_panel_thwc(video_01: np.ndarray, n_frames: int, h: int, w: int, label: str) -> np.ndarray:
        """[T',H',W',3] (or [1,H',W',3] for a static image) float32 in [0,1] -> [n_frames,h,w,3], nearest
        temporal resample + bilinear spatial resize, with `label` burned into the top-left corner (cv2,
        no external font file) so a stacked/concatenated video stays self-describing without a separate
        legend. Common panel builder for the wan-input / vjepa-input / pred / gt (/ donor-gt) stack that
        every validate_step call (suite and normal) now logs, per the project convention of labelling
        input/output panels explicitly."""
        t_src = video_01.shape[0]
        idx = np.clip((np.arange(n_frames) * t_src) // max(n_frames, 1), 0, t_src - 1)
        frames = video_01[idx]  # [n_frames,H',W',3]
        out = np.stack([
            cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames
        ]).astype(np.float32)
        out = out.clip(0, 1)
        for f in out:
            # Draw twice (thick black outline, thin yellow fill) so the label reads on both dark and
            # light content -- cv2 draws in-place regardless of RGB/BGR channel order for plain text.
            cv2.putText(f, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.0, 0.0, 0.0), 3, cv2.LINE_AA)
            cv2.putText(f, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1.0, 1.0, 0.0), 1, cv2.LINE_AA)
        return out

    def _wan_input_panels(self, sample, i, first_01_i) -> list:
        """[(label, image_hwc_01)] for the WAN-INPUT side of the stacked viz / wan_input_image
        preview -- user request: "save bursts with 3 and 5 frames as well [not] only single image.
        If images > 1, stack the input images ... to visualize all bursts separately." One panel per
        burst segment when burst_size > 1 (all of them, not just the first/accessible one), else the
        single accessible conditioning image unchanged. `first_01_i` is [3,H,W] in [0,1] (the existing
        single-image fallback, computed once by the caller from `_first_frame_condition`)."""
        if self.burst_size > 1 and isinstance(sample, dict) and "burst_blur" in sample:
            burst01 = ((sample["burst_blur"][i].float() + 1.0) / 2.0).clip(0, 1).cpu().numpy()  # [S,3,H,W]
            return [(f"WAN INPUT SEG{s}", burst01[s].transpose(1, 2, 0)) for s in range(burst01.shape[0])]
        return [("WAN INPUT", first_01_i.transpose(1, 2, 0))]

    def _stacked_input_output_viz(
        self, wan_input_panels, vjepa_input_01, gen_thwc, gt_thwc, donor_gt_01=None,
    ) -> np.ndarray:
        """Builds the ONE stacked visualization video every validate_step call now logs (suite and
        normal alike), in ADDITION to the separate wan_output_/gt_/etc. clips already saved --
        left-to-right: one WAN INPUT panel per burst segment (see _wan_input_panels) | VJEPA INPUT
        (the actual pixels V-JEPA/the predictor saw -- a placeholder panel for a token-only
        intervention that only perturbs the grid, e.g. shuffle/zero/reverse, since those never touch
        pixels) | PRED (Wan's output) | GT (the target clip's own ground truth), plus a 5th DONOR GT
        panel when swap_donor is the active arm, so a donor-following claim can be checked by eye
        against the donor's own real motion (see validate_step / _apply_token_arm's swap_donor case
        for why the scalar correlation-vs-donor-GT metric alone can miss a real transfer here).
        wan_input_panels: [(label, [H,W,3] float32 in [0,1]), ...] from _wan_input_panels -- each gets
        its OWN full-width (w) tile, so a multi-segment burst is shown side by side, not squeezed into
        one tile. vjepa_input_01 [T',H',W',3] in [0,1] or None; gen_thwc/gt_thwc [T,h,w,3] in [0,1]
        (the already-cropped/aligned pair used for scoring); donor_gt_01 [T'',H'',W'',3] or None.
        Returns [T,h,tile_w*n_panels,3] float32 in [0,1]."""
        n_frames, h, w = gt_thwc.shape[0], gt_thwc.shape[1], gt_thwc.shape[2]
        panels = [self._resample_panel_thwc(img[np.newaxis], n_frames, h, w, label) for label, img in wan_input_panels]
        if vjepa_input_01 is not None:
            panels.append(self._resample_panel_thwc(vjepa_input_01, n_frames, h, w, "VJEPA INPUT"))
        else:
            panels.append(self._resample_panel_thwc(
                np.zeros((1, h, w, 3), dtype=np.float32), n_frames, h, w, "VJEPA INPUT (token-only arm)"))
        panels.append(self._resample_panel_thwc(gen_thwc, n_frames, h, w, "PRED"))
        panels.append(self._resample_panel_thwc(gt_thwc, n_frames, h, w, "GT"))
        if donor_gt_01 is not None:
            panels.append(self._resample_panel_thwc(donor_gt_01, n_frames, h, w, "DONOR GT"))
        return np.concatenate(panels, axis=2)

    @staticmethod
    def _draw_flow_arrows(image_01: np.ndarray, flow_2hw: np.ndarray, step: int = 16,
                           color=(255, 0, 0), scale: float = 1.0) -> np.ndarray:
        """image_01: [H,W,3] float in [0,1]. flow_2hw: [2,H,W] (dx, dy). Returns a uint8 [H,W,3] copy with arrows drawn."""
        img = (image_01.clip(0, 1) * 255).astype(np.uint8).copy()
        h, w = flow_2hw.shape[1:]
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                dx = float(flow_2hw[0, y, x]) * scale
                dy = float(flow_2hw[1, y, x]) * scale
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    continue
                cv2.arrowedLine(img, (x, y), (int(round(x + dx)), int(round(y + dy))), color, 1, tipLength=0.35)
        return img

    # --- training ---

    def _maybe_autocast(self):
        """bf16 autocast iff some transformer params were promoted to fp32 (unfreeze_modules); otherwise a no-op context."""
        if self._unfreeze_fp32:
            return torch.autocast(device_type="cuda", dtype=self.weight_dtype)
        import contextlib
        return contextlib.nullcontext()

    def _sample_t(self, bsz, device):
        """Training noise level t in (0,1) per sample. See `t_sampling` in __init__."""
        if self.t_sampling == "uniform":
            t = torch.rand(bsz, device=device, dtype=torch.float32)
        else:
            if self.t_sampling == "logit_normal":
                u = torch.sigmoid(torch.randn(bsz, device=device, dtype=torch.float32) * self.t_logit_std + self.t_logit_mean)
            else:
                u = torch.rand(bsz, device=device, dtype=torch.float32)
            sh = self.t_shift
            t = (sh * u / (1.0 + (sh - 1.0) * u)).clamp(1e-4, 1.0 - 1e-4)
        if self.t_low_frac > 0:
            low = torch.rand(bsz, device=device) < self.t_low_frac
            t = torch.where(low, torch.rand(bsz, device=device, dtype=torch.float32) * self.t_low_max, t)
        return t

    def _perturb_tokens(self, grid):
        """TRAIN-only oracle-token corruption (token_dropout / token_noise_std). grid [B, N, D*K]."""
        if self.token_dropout <= 0 and self.token_noise_std <= 0:
            return grid
        g = grid
        if self.token_noise_std > 0:
            d = self.vjepa_feature_dim
            chunks = []
            for c in g.split(d, dim=-1):
                rms = c.float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                chunks.append(c + torch.randn_like(c) * rms * self.token_noise_std)
            g = torch.cat(chunks, dim=-1)
        if self.token_dropout > 0:
            keep = (torch.rand(g.shape[0], g.shape[1], 1, device=g.device) >= self.token_dropout).to(g.dtype)
            g = g * keep
        return g

    def optimize_parameters(self):
        sample = self.sample
        wan_frames = sample["wan_frames"].to(self.accelerator.device, dtype=torch.float32)  # [B,T,3,H,W] in [-1,1]
        bsz, T, _, H, W = wan_frames.shape
        # Prepend deblur_prefix_frames copies of the real first frame (no-op
        # when 0) -- gives the model a dedicated priming region whose
        # target is always "the true sharp first frame," before the frames
        # requiring real motion prediction. See _prepend_deblur_frames.
        wan_frames_extended = self._build_target_video(sample, wan_frames)
        video_bcthw = wan_frames_extended.permute(0, 2, 1, 3, 4)  # [B,3,T',H,W] -- the sharp clip, denoising target only
        # The I2V first-frame condition must be the actual accessible input
        # (the blurred/frame-averaged image) -- Adobe240Dataset applies the
        # same crop to `blur`/`burst_blur` and `wan_frames`, so shapes
        # already match. Using wan_frames' own first frame here would leak a
        # sharp frame from the target clip into the conditioning signal.
        first_frame_bchw = self._first_frame_condition(sample)  # [B,3,H,W] in [-1,1]

        with torch.no_grad():
            x0 = self._encode_video_sample(video_bcthw)  # [B,z,T_lat,H_lat,W_lat]
            latent_condition = self._encode_first_frame_argmax(first_frame_bchw)  # [B,z,1,H_lat,W_lat]
        x0 = x0.to(torch.float32)
        latent_condition = latent_condition.to(torch.float32)
        _, z_dim, t_lat, h_lat, w_lat = x0.shape

        first_frame_mask = torch.ones(1, 1, t_lat, h_lat, w_lat, device=x0.device, dtype=torch.float32)
        first_frame_mask[:, :, 0] = 0

        noise = torch.randn_like(x0)
        t = self._sample_t(bsz, x0.device)  # [B] in (0,1)
        t_view = t.view(bsz, 1, 1, 1, 1)
        x_t_interp = (1 - t_view) * x0 + t_view * noise
        latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * x_t_interp
        velocity_target = noise - x0

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = t_lat // p_t
        post_patch_height = h_lat // p_h
        post_patch_width = w_lat // p_w

        num_train_timesteps = getattr(self.scheduler.config, "num_train_timesteps", 1000)
        # Built directly at the post-patch shape rather than via strided
        # slicing of first_frame_mask (`::p_h, ::p_w`), which silently
        # diverges from patch_embedding's floor-based conv downsampling
        # whenever h_lat/w_lat is odd (ceil-vs-floor off-by-one) -- only
        # exposed once a non-320-square resolution made an odd latent
        # dimension possible. The mask is uniform across every spatial
        # position within a given temporal frame (0 at the first latent
        # frame, 1 elsewhere), so direct construction is always correct.
        mask_strided = torch.ones(
            post_patch_num_frames, post_patch_height, post_patch_width, device=x0.device, dtype=torch.float32,
        )
        mask_strided[0] = 0
        timestep_2d = mask_strided.unsqueeze(0) * (t * num_train_timesteps).view(bsz, 1, 1, 1)
        timestep_2d = timestep_2d.reshape(bsz, -1)  # [B, seq_len]

        dynamics_grid, _ = self._dynamics_full_grid(sample)
        if self.disable_dynamics_injection:
            # Capacity-matched ablation: zero the V-JEPA signal but keep it
            # flowing through the connector/PGA/modulation modules (instead
            # of disabling the injection path outright), so those modules
            # stay trainable -- with lora_finetune=false they are the only
            # trainable parameters, so disabling the path entirely leaves
            # nothing with a grad_fn and loss.backward() raises.
            dynamics_grid = torch.zeros_like(dynamics_grid)
        else:
            dynamics_grid = self._perturb_tokens(dynamics_grid)
        # Only the value-residual path consumes this. Under cross_attention it
        # was previously computed and then thrown away -- not merely wasted
        # work: pool_dynamics_to_frame_tokens asserts the grid has exactly
        # self.vjepa_num_tubelets tubelets, which under dynamic_burst is the
        # MAXIMUM (20), so every step with S < max raised
        #   "got N=2304 tokens, expected ...=11520".
        # Validation never caught it because val is pinned at S=max.
        dynamics_full = None
        if self.dynamics_injection != "cross_attention":
            dynamics_full = self._dynamics_tokens_with_deblur_padding(
                dynamics_grid, post_patch_num_frames, post_patch_height, post_patch_width
            )

        ctx = self.dynamics_context
        ctx.enabled = True
        ctx.t = t
        if self.dynamics_injection == "cross_attention":
            # Clean swap, not both paths at once.
            ctx.dynamics_tokens = None
            kv = self._build_dynamics_kv(
                dynamics_grid, post_patch_num_frames, post_patch_height, post_patch_width
            )
            ctx.dynamics_kv = ([k.to(self.weight_dtype) for k in kv] if isinstance(kv, (list, tuple)) else kv.to(self.weight_dtype))
        else:
            ctx.dynamics_tokens = dynamics_full.to(self.weight_dtype)
            ctx.dynamics_kv = None
        blur_tokens = self._burst_blur_condition_tokens(sample)
        if blur_tokens is None and self.blur_every_frame:
            blur_tokens = self.blur_cond_embed(latent_condition.float().expand(-1, -1, t_lat, -1, -1))
        ctx.blur_tokens = blur_tokens.to(self.weight_dtype) if blur_tokens is not None else None

        prompt_embeds = self._prompt_embeds(bsz)

        with self._maybe_autocast():
            model_output = self.transformer(
                hidden_states=latent_model_input.to(self.weight_dtype),
                timestep=timestep_2d.to(self.weight_dtype),
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        # Masked flow-matching loss: frame-0's latent is hardwired to the
        # clean condition regardless of t (never mixes with `noise`), so
        # `velocity_target` there is an arbitrary, unobservable target --
        # leaving it unmasked injects pure-variance gradient with no
        # learnable signal.
        _lw = self._latent_loss_weights(t_lat, wan_frames.shape[1], x0.device)
        if not getattr(self, "_layout_logged", False):
            self._layout_logged = True
            self.logger.info(
                "LAYOUT: exposure T=%d, front=%d, back=%d -> target video %d px frames -> latent %s -> post-patch (t,h,w)=(%d,%d,%d)=%d tokens; "
                "per-latent loss weights (latent 0 = condition, masked) %s",
                wan_frames.shape[1], self.ctx_front, self.ctx_back, wan_frames_extended.shape[1], tuple(x0.shape),
                post_patch_num_frames, post_patch_height, post_patch_width,
                post_patch_num_frames * post_patch_height * post_patch_width, [round(float(v), 2) for v in _lw.flatten()])
        frame_mask = (first_frame_mask * _lw).expand_as(model_output)
        per_elem = F.mse_loss(model_output.float(), velocity_target.float(), reduction="none")
        loss = (per_elem * frame_mask).sum() / frame_mask.sum().clamp(min=1)
        wan_loss, aux = loss, None
        if self.predictor_aux_weight > 0 and self.dynamics_source == "blur_jepa":
            # anchor the (trainable) predictor to the ORACLE tokens so Wan's loss cannot pull it arbitrarily far
            with torch.no_grad():
                oracle_grid, _ = self._oracle_grid_burst(sample)
            aux = F.mse_loss(dynamics_grid.float(), oracle_grid.float())
            loss = loss + self.predictor_aux_weight * aux

        reblur = pixel = None
        if self.reblur_weight > 0 or self.pixel_loss_weight > 0:
            reblur, pixel = self._decoded_x0_losses(
                model_output, x_t_interp, t, latent_condition, first_frame_bchw, wan_frames, sample)
            if reblur is not None and self.reblur_weight > 0:
                loss = loss + self.reblur_weight * reblur
            if pixel is not None and self.pixel_loss_weight > 0:
                loss = loss + self.pixel_loss_weight * pixel

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self._trainable_parameters(), getattr(self.opt.train.optim, "max_grad_norm", 1.0)
            )
            self.optimizers[0].step()
            self.optimizers[0].zero_grad()

        out = {"all": loss.detach()}
        if aux is not None:
            out["wan"], out["aux"] = wan_loss.detach(), aux.detach()
        if reblur is not None:
            out["reblur"] = reblur.detach()
        if pixel is not None:
            out["pixel"] = pixel.detach()
        return out

    def _get_lpips(self):
        if self._lpips_net is None:
            import lpips
            self._lpips_net = lpips.LPIPS(net="alex", verbose=False).to(self.accelerator.device).eval().requires_grad_(False)
        return self._lpips_net

    def _decoded_x0_losses(self, v_pred, x_t, t, latent_condition, first_frame_bchw, wan_frames, sample):
        """Losses on the DECODED x0_hat = x_t - t * v_pred (flow matching: x_t=(1-t)x0+t*noise, v=noise-x0). Returns (reblur, pixel);
        either is None when not applicable this step.
          reblur: mean over the F real generated frames (pixel space) vs the target. reblur_target='blur' compares to the INPUT blur,
                  which is the mean of the FULL exposure window -- only valid when the clip spans that window (F == window frames);
                  'gt_mean' compares to the mean of the GT clip's own F frames and is exposure-consistent for any F (use it with F<9).
          pixel:  L1 and/or LPIPS between the generated and GT CENTER frame (time-reversal invariant for odd F).
        Only samples with t < reblur_t_max / pixel_t_max (x0_hat is meaningful there; at high noise it is a blurry mean).
        Frame-0 latent is the clean condition exactly as in inference; the deblur prefix is cropped like validation. One VAE decode
        (gradient-checkpointed) is shared by both terms."""
        want_reblur = self.reblur_weight > 0 and not (self.burst_size != 1 and self.reblur_target == "blur")
        want_pixel = self.pixel_loss_weight > 0
        n_px = first_frame_bchw.shape[-1] * first_frame_bchw.shape[-2]
        if want_reblur and self.reblur_max_pixels and n_px > self.reblur_max_pixels:
            want_reblur = False
        if want_pixel and self.pixel_max_pixels and n_px > self.pixel_max_pixels:
            want_pixel = False
        if not (want_reblur or want_pixel):
            return None, None
        t_max = max(self.reblur_t_max if want_reblur else 0.0, self.pixel_t_max if want_pixel else 0.0)
        sel = (t < t_max).nonzero(as_tuple=True)[0][: self.reblur_frames]
        if sel.numel() == 0:
            return None, None
        tv = t[sel].view(-1, 1, 1, 1, 1)
        x0_hat = x_t[sel].float() - tv * v_pred[sel].float()
        x0_hat = torch.cat([latent_condition[sel].float(), x0_hat[:, :, 1:]], dim=2)   # frame 0 = clean condition

        def dec(z):
            # bf16 autocast halves the VAE decode+backward activation memory (full-res fp32 decode with grad OOM'd an 80GB A100 at 512^2)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._decode_latents(z.to(self.weight_dtype))
        video = torch.utils.checkpoint.checkpoint(dec, x0_hat, use_reentrant=False)       # [b,3,T',H,W] in [-1,1]
        video = self._crop_exposure(video, wan_frames.shape[1]).float()
        reblur = pixel = None
        if want_reblur:
            ok = (t[sel] < self.reblur_t_max)
            if bool(ok.any()):
                mean_pred = video[ok].mean(dim=2)                                          # [b,3,H,W]
                if self.reblur_target == "gt_mean":
                    target = wan_frames[sel][ok].float().mean(dim=1)
                else:
                    target = first_frame_bchw[sel][ok].float()
                reblur = F.mse_loss(mean_pred, target) if self.reblur_loss_type == "mse" else (mean_pred - target).abs().mean()
        if want_pixel:
            ok = (t[sel] < self.pixel_t_max)
            if bool(ok.any()):
                ci = video.shape[2] // 2
                pred_c = video[ok][:, :, ci]                                               # [b,3,H,W] in [-1,1]
                gt_c = wan_frames[sel][ok][:, ci].float()                                  # wan_frames is [B,T,3,H,W] pm1
                terms = []
                if "l1" in self.pixel_loss_type:
                    terms.append((pred_c - gt_c).abs().mean())
                if "lpips" in self.pixel_loss_type:
                    terms.append(self._get_lpips()(pred_c, gt_c).mean())
                pixel = sum(terms)
        return reblur, pixel

    # --- validation ---

    @torch.no_grad()
    def validation(self):
        if self.suite is not None and not self.is_train:
            return self._run_suite()
        torch.cuda.empty_cache()
        for model in self.models:
            model.eval()

        # dynamic_burst.val_sizes: run this SAME validation pass once per requested burst size
        # (e.g. [3, 5]) instead of the single size training validation used to be pinned at, so a
        # burst-length-specific regression shows up during training. Single-value configs (or no
        # dynamic_burst at all) run exactly one pass as before -- this is purely additive.
        val_sizes = self.dyn_burst["val_sizes"] if self.dyn_burst is not None else [None]
        self._val_acc = {"epe": [], "psnr": [], "ssim": []}
        for S in val_sizes:
            if S is not None:
                self.dyn_burst["val_size"] = S
            self._val_stem_suffix = f"_burst{S}" if len(val_sizes) > 1 else ""
            idx = 0
            desc = f"Validation (burst={S})" if len(val_sizes) > 1 else "Validation"
            for batch in tqdm(self.test_dataloader, desc=desc):
                idx = self.validate_step(batch, idx)
                if idx >= self.max_val_steps:
                    break
        self._val_stem_suffix = ""
        # clip-averaged metrics of this whole validation pass (base_model._update_best_checkpoint reads them for save_best)
        self.last_val_metrics = {k: float(np.mean(v)) for k, v in self._val_acc.items() if v}
        if self.last_val_metrics:
            log_metric(self.accelerator, {f"val_mean/{k}": v for k, v in self.last_val_metrics.items()}, self.global_step)
            self.logger.info("validation mean over %d clips: %s", len(self._val_acc["psnr"]),
                             ", ".join(f"{k}={v:.4f}" for k, v in self.last_val_metrics.items()))

        for model in self.models:
            model.train()
        torch.cuda.empty_cache()

    def _dino_tokens(self, clip01):
        """clip01 [B,T,3,H,W] in [0,1] -> [B, (T//2)*576, 1024] frozen DINOv2-large tokens (see dynamics_encoder_type)."""
        if self.dino is None:
            from transformers import AutoModel
            self.dino = AutoModel.from_pretrained("facebook/dinov2-large").to(
                self.accelerator.device, dtype=self.vjepa_dtype).eval().requires_grad_(False)
            self.logger.info("loaded frozen DINOv2-large as the dynamics encoder (P2c)")
        B, T = clip01.shape[:2]
        x = clip01.reshape(B * T, *clip01.shape[2:]).float()
        x = torch.nn.functional.interpolate(x, size=(336, 336), mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.vjepa_dtype)
        feats = self.dino(pixel_values=x).last_hidden_state[:, 1:]          # drop CLS -> [B*T, 576, 1024]
        feats = feats.reshape(B, T // 2, 2, feats.shape[1], feats.shape[2]).mean(dim=2)   # pair-average = tubelet 2
        n_t, cells, dim = feats.shape[1], feats.shape[2], feats.shape[3]                 # [B, T//2, 576, 1024]
        return feats.reshape(B, n_t * cells, dim).float()

    # ---------------- intervention suite ----------------

    def _run_suite(self):
        import json
        torch.cuda.empty_cache()
        for model in self.models:
            model.eval()
        base_root = self.opt.path.experiments_root
        trackers_backup = list(self.accelerator.trackers)
        self.accelerator.trackers = []  # no per-arm Comet uploads (thousands of videos)
        try:
            for arm in self.suite["arms"]:
                for sd in self.suite["seeds"]:
                    tag = f"{arm['name']}_s{sd}"
                    outdir = os.path.join(base_root, tag)
                    if os.path.exists(os.path.join(outdir, "DONE.json")):
                        self.logger.info("suite: skip %s (already done)", tag)
                        continue
                    self.opt.path.experiments_root = outdir
                    self.sampler_seed = int(sd)
                    self._clip_stems, self._donor_log = [], []
                    self._enter_arm(arm)
                    try:
                        idx = 0
                        for batch in tqdm(self.test_dataloader, desc=f"suite {tag}"):
                            idx = self.validate_step(batch, idx)
                            if idx >= self.max_val_steps:
                                break
                    finally:
                        self._exit_arm()
                    os.makedirs(outdir, exist_ok=True)
                    with open(os.path.join(outdir, "DONE.json"), "w") as f:
                        json.dump({"arm": arm, "seed": int(sd), "stems": self._clip_stems,
                                   "donor": self._donor_log, "n": int(idx)}, f)
                    self.logger.info("suite: finished %s (%d clips)", tag, idx)
        finally:
            self.opt.path.experiments_root = base_root
            self.accelerator.trackers = trackers_backup
            for model in self.models:
                model.train()
            torch.cuda.empty_cache()

    def _enter_arm(self, arm):
        """Apply a MODULE intervention (token interventions are applied per clip in _apply_token_arm)."""
        self.active_arm = arm
        self._arm_undo = []
        mode = arm["mode"]
        xa = self.dynamics_xattn
        if mode == "xattn_off":
            for m in xa:
                self._arm_undo.append((m.to_out.weight, m.to_out.weight.detach().clone()))
                m.to_out.weight.data.zero_()
        elif mode in ("pos_t_shuffle", "pos_s_shuffle", "pos_zero"):
            n_used = int(self.vjepa_num_tubelets)
            for bi, m in enumerate(xa):
                g = torch.Generator(device="cpu").manual_seed(1234 + bi)
                if mode == "pos_t_shuffle":
                    self._arm_undo.append((m.pos_t, m.pos_t.detach().clone()))
                    perm = torch.randperm(n_used, generator=g).to(m.pos_t.device)
                    m.pos_t.data[:n_used] = m.pos_t.data[:n_used][perm]
                elif mode == "pos_s_shuffle":
                    self._arm_undo.append((m.pos_s, m.pos_s.detach().clone()))
                    perm = torch.randperm(m.pos_s.shape[0], generator=g).to(m.pos_s.device)
                    m.pos_s.data.copy_(m.pos_s.data[perm])
                else:
                    self._arm_undo.append((m.pos_t, m.pos_t.detach().clone()))
                    self._arm_undo.append((m.pos_s, m.pos_s.detach().clone()))
                    m.pos_t.data.zero_()
                    m.pos_s.data.zero_()
        elif mode == "lora_off":
            self._lora_ctx = self.transformer.disable_adapter()   # peft context manager
            self._lora_ctx.__enter__()

    def _exit_arm(self):
        for param, orig in self._arm_undo:
            param.data.copy_(orig)
        self._arm_undo = []
        if self._lora_ctx is not None:
            self._lora_ctx.__exit__(None, None, None)
            self._lora_ctx = None
        self.active_arm = None

    def _apply_token_arm(self, grid, idx):
        """grid [B, N, D] (N = T*cells, flattened (t,h,w)) -> perturbed grid. Positional structure of the KV is
        untouched (only the CONTENT moves), so a temporal intervention tests whether the model reads ORDER."""
        arm = self.active_arm
        mode = arm["mode"] if arm else None
        if mode in (None, "none", "cue_gt", "cue_gt_flip", "stroke_fwd", "stroke_rev", "stroke_shuffle", "stroke_none", "edit_keep", "edit_gt_fwd", "edit_gt_rev", "edit_gt_region_rev", "edit_json", "xattn_off", "lora_off", "pos_t_shuffle", "pos_s_shuffle", "pos_zero",
                    "input_reverse", "input_reverse_blur", "blur_reverse", "token_blend", "axis_oracle", "oracle_axis_B",
                    "input_reverse_within_seg", "input_shuffle", "input_shuffle_global"):
            return grid
        B, N, D = grid.shape
        cells = self.vjepa_grid_h * self.vjepa_grid_w
        T = N // cells
        P = int(self.per_segment_vjepa_num_tubelets)
        S = max(T // P, 1)
        g = grid.reshape(B, T, cells, D)
        gen = torch.Generator(device="cpu").manual_seed(int(self.sampler_seed or 0) * 7919 + int(idx) * 31 + 17)
        if mode == "zero":
            return torch.zeros_like(grid)
        if mode == "reverse_within_seg":
            g = g.reshape(B, S, P, cells, D).flip(2).reshape(B, T, cells, D)
        elif mode == "resample_time":
            # Compress (scale<1) / expand (scale>1) the token time axis; the Wan output length is set to match
            # in validate_step via _resample_out_frames (cross-attention treats the tokens as a set, so T_J != T_W is fine).
            t_new = max(1, int(round(float(arm["scale"]) * T)))
            gt_ = g.permute(0, 2, 3, 1).reshape(B, cells * D, T)
            gt_ = F.interpolate(gt_.float(), size=t_new, mode="linear", align_corners=True).to(g.dtype)
            g = gt_.reshape(B, cells, D, t_new).permute(0, 3, 1, 2)
            return g.reshape(B, t_new * cells, D)
        elif mode == "shift_segments":
            g = g.reshape(B, S, P, cells, D).roll(1, dims=1).reshape(B, T, cells, D)
        elif mode == "shuffle_tubelets":
            g = g[:, torch.randperm(T, generator=gen).to(g.device)]
        elif mode == "shuffle_space":
            g = g[:, :, torch.randperm(cells, generator=gen).to(g.device)]
        elif mode == "noise":
            noise = torch.randn(g.shape, generator=gen).to(g.device, g.dtype)
            g = g + float(arm["sigma"]) * g.float().std().to(g.dtype) * noise
        elif mode == "keep_seg":
            seg = int(arm["seg"])
            gg = g.float().mean(dim=(1, 2), keepdim=True).to(g.dtype).expand_as(g).clone()   # neutral filler
            gg[:, seg * P:(seg + 1) * P] = g[:, seg * P:(seg + 1) * P]
            g = gg
        elif mode == "swap_donor":
            # Donor = the dynamics of ANOTHER clip, from a DIFFERENT SCENE (MixedDataset shuffles its plan, so a
            # fixed index offset can land in the same scene and make the swap meaningless). Start at idx+offset
            # and step forward to the first clip whose scene differs from the target's.
            import re as _re
            def _stem(x):
                return os.path.splitext(os.path.basename(x))[0]
            def _scene(st):
                s2, n = _re.subn(r"_c\d+_burst\d+$", "", st)
                return s2 if n else _re.sub(r"_\d+$", "", st)
            ds = self.test_dataloader.dataset
            nm = self.sample.get("name") if isinstance(self.sample, dict) else None
            cur = _stem(nm[0]) if nm else None
            j = d_stem = donor = None
            for step in range(len(ds)):
                cand = (int(idx) + int(arm["offset"] or 1) + step) % len(ds)
                if cand == int(idx):
                    continue
                donor = ds.collate_fn([ds[cand]])
                dn = donor.get("name")
                d_stem = _stem(dn[0]) if dn else None
                if cur is None or d_stem is None or _scene(d_stem) != _scene(cur):
                    j = cand
                    break
            if j is None:
                raise RuntimeError("swap_donor: no clip from a different scene found")
            dgrid, dvjepa01 = self._dynamics_full_grid(donor)
            self._donor_log.append([int(idx), int(j), d_stem])
            # Stash for the stacked visualization (validate_step reads this right after calling us):
            # the donor's own V-JEPA input (what's now actually driving the tokens) and its own GT
            # video, so "does the output follow the donor" can be checked by eye against the donor's
            # real motion, not just the target's -- the conflict this exists to help resolve is that
            # the SCALAR moving-content-correlation-vs-donor-GT metric can read ~0 even when the output
            # visibly follows the donor's motion, because donor/target are each cropped around their
            # OWN peak-motion location (motion_square) -- a correctly-transferred motion would land in
            # a different part of the frame than the donor's own GT, which a pixel-position correlation
            # cannot see. This does not settle that by itself; it just makes the eye check possible.
            self._donor_viz = {
                "stem": d_stem, "vjepa01": dvjepa01,
                "gt01": ((donor["wan_frames"].to(self.accelerator.device).float() + 1.0) / 2.0)
                if "wan_frames" in donor else None,
            }
            return dgrid
        else:
            raise ValueError(f"unknown intervention mode {mode!r}")
        return g.reshape(B, N, D)

    def _resample_out_frames(self, t_gt: int, scale: float) -> int:
        """Output pixel-frame count for a token-time scale: duration scales with `scale` ((F-1) -> scale*(F-1)),
        snapped to the nearest count Wan can produce, i.e. (F + deblur_prefix - 1) % vae_scale_factor_temporal == 0."""
        r = int(self.vae_scale_factor_temporal)
        target = scale * (t_gt - 1) + 1
        fb = self.ctx_front + self.ctx_back
        k = int(round((target + fb - 1) / r))
        f = k * r - (fb - 1)
        while f < 1:
            f += r
        return int(f)

    @torch.no_grad()
    def validate_step(self, batch, idx):
        self.feed_data(batch, is_train=False)
        self._cur_val_idx = int(idx)   # seeds the S=1 stroke control deterministically per clip
        sample = self.sample
        _bm = self.active_arm["mode"] if (self.suite is not None and self.active_arm) else None
        if _bm in ("blur_reverse", "input_reverse_blur") and isinstance(sample, dict) and "burst_blur" in sample:
            # Reversed WORLD: the blurred burst is fed in reverse segment order too (first-frame anchor = old LAST blur).
            # blur_reverse = blur order only (tokens forward); input_reverse_blur = V-JEPA input reversed AND blur order reversed.
            sample = dict(sample); sample["burst_blur"] = sample["burst_blur"].flip(1); self.sample = sample
        device = self.accelerator.device
        num_inference_steps = int(_cfg(self.opt.val, "num_inference_steps", 50))

        wan_frames = sample["wan_frames"].to(device, dtype=torch.float32)  # [B,T,3,H,W] in [-1,1]
        bsz, T, _, H, W = wan_frames.shape
        # The I2V first-frame condition is the actual accessible input (the
        # blurred image, or burst mode's first segment), not a sharp frame
        # from the target clip -- see the matching comment in
        # optimize_parameters.
        first_frame_bchw = self._first_frame_condition(sample)  # [B,3,H,W] in [-1,1]

        latent_condition = self._encode_first_frame_argmax(first_frame_bchw).to(torch.float32)
        z_dim = self.vae.config.z_dim
        h_lat = H // self.vae_scale_factor_spatial
        w_lat = W // self.vae_scale_factor_spatial
        # Generate at the deblur-extended length (no-op when
        # deblur_prefix_frames == 0) -- the extra frames get cropped off
        # after decoding, before any GT comparison/logging below.
        # Dynamics tokens (+ intervention arm) are built BEFORE the latent length is fixed: the resample_time
        # arm changes how many frames Wan must generate (see _resample_out_frames).
        dynamics_grid, vjepa_input_clip_01 = self._dynamics_full_grid(sample)
        if self.disable_dynamics_injection:
            dynamics_grid = torch.zeros_like(dynamics_grid)
        self._donor_viz = None
        if self.suite is not None and self.active_arm is not None:
            dynamics_grid = self._apply_token_arm(dynamics_grid, idx)
        dynamics_grid = self._apply_editor(dynamics_grid, sample, idx)
        dynamics_grid = self._apply_blend(dynamics_grid, sample, idx)
        T_gt = T
        self._out_frames_override = None
        if self.suite is not None and self.active_arm is not None and self.active_arm["mode"] == "resample_time":
            T = self._resample_out_frames(T_gt, float(self.active_arm["scale"]))
            self._out_frames_override = T
        t_lat = (T + self.ctx_front + self.ctx_back - 1) // self.vae_scale_factor_temporal + 1

        first_frame_mask = torch.ones(1, 1, t_lat, h_lat, w_lat, device=device, dtype=torch.float32)
        first_frame_mask[:, :, 0] = 0

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = t_lat // p_t
        post_patch_height = h_lat // p_h
        post_patch_width = w_lat // p_w
        # See the matching comment in optimize_parameters: built directly at
        # the post-patch shape rather than via strided slicing, which
        # diverges from patch_embedding's floor-based downsampling for odd
        # h_lat/w_lat.
        mask_strided = torch.ones(
            post_patch_num_frames, post_patch_height, post_patch_width, device=device, dtype=torch.float32,
        )
        mask_strided[0] = 0

        if self._donor_viz is not None and self._donor_viz.get("vjepa01") is not None:
            # swap_donor: what's now actually driving the cross-attention is the DONOR's V-JEPA input,
            # not the target's -- show that in the stacked viz instead of the (now-irrelevant) target one.
            vjepa_input_clip_01 = self._donor_viz["vjepa01"]
        # Only the value-residual path consumes this. Under cross_attention it
        # was previously computed and then thrown away -- not merely wasted
        # work: pool_dynamics_to_frame_tokens asserts the grid has exactly
        # self.vjepa_num_tubelets tubelets, which under dynamic_burst is the
        # MAXIMUM (20), so every step with S < max raised
        #   "got N=2304 tokens, expected ...=11520".
        # Validation never caught it because val is pinned at S=max.
        dynamics_full = None
        if self.dynamics_injection != "cross_attention":
            dynamics_full = self._dynamics_tokens_with_deblur_padding(
                dynamics_grid, post_patch_num_frames, post_patch_height, post_patch_width
            )
        ctx = self.dynamics_context
        ctx.enabled = True
        if self.dynamics_injection == "cross_attention":
            # Clean swap, not both paths at once.
            ctx.dynamics_tokens = None
            kv = self._build_dynamics_kv(
                dynamics_grid, post_patch_num_frames, post_patch_height, post_patch_width
            )
            ctx.dynamics_kv = ([k.to(self.weight_dtype) for k in kv] if isinstance(kv, (list, tuple)) else kv.to(self.weight_dtype))
        else:
            ctx.dynamics_tokens = dynamics_full.to(self.weight_dtype)
            ctx.dynamics_kv = None
        blur_tokens = self._burst_blur_condition_tokens(sample)
        if blur_tokens is None and self.blur_every_frame:
            blur_tokens = self.blur_cond_embed(latent_condition.float().expand(-1, -1, t_lat, -1, -1))
        ctx.blur_tokens = blur_tokens.to(self.weight_dtype) if blur_tokens is not None else None

        prompt_embeds = self._prompt_embeds(bsz)
        num_train_timesteps = getattr(self.scheduler.config, "num_train_timesteps", 1000)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        _gen = None
        if self.sampler_seed is not None:
            # per-(seed, sample index): reproducible, differs across seeds, and identical for two arms
            # given the same seed and clip (common random numbers)
            _gen = torch.Generator(device=device).manual_seed(self.sampler_seed * 100003 + int(idx))
        latents = torch.randn(bsz, z_dim, t_lat, h_lat, w_lat, device=device, dtype=torch.float32, generator=_gen)

        for t_raw in self.scheduler.timesteps:
            latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
            ctx.t = (t_raw.float() / num_train_timesteps).expand(bsz)

            timestep_2d = mask_strided.unsqueeze(0) * t_raw.float()
            timestep_2d = timestep_2d.expand(bsz, -1, -1, -1).reshape(bsz, -1)

            with self._maybe_autocast():
                noise_pred = self.transformer(
                    hidden_states=latent_model_input.to(self.weight_dtype),
                    timestep=timestep_2d.to(self.weight_dtype),
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            latents = self.scheduler.step(noise_pred.float(), t_raw, latents, return_dict=False)[0]

        latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
        video_with_prefix = self._decode_latents(latents)  # [B,3,T+deblur_prefix_frames,H,W] in [-1,1]
        # Crop the deblur-priming frames off before any GT comparison/logging
        # below -- they have no real GT beyond "repeat of frame 0" and exist
        # only to prime the model's implicit deblurring, not to be scored
        # or compared as generated motion content. No-op when
        # deblur_prefix_frames == 0.
        video = self._crop_exposure(video_with_prefix, T)  # [B,3,T,H,W] in [-1,1]

        # Optical flow (validation-time visualization only): computed separately
        # for GT and pred, both as a per-frame flow video and as a single
        # first-to-last-frame displacement field for the arrow overlay.
        video_btchw = video.permute(0, 2, 1, 3, 4)  # [B,T,3,H,W]
        if self.suite is None:   # suite runs only need the GT/generated videos (scored offline)
            gt_flows = self._resize_flow_to(self._flow_from_video(wan_frames), H, W)  # [B,T,2,H,W]
            pred_flows = self._resize_flow_to(self._flow_from_video(video_btchw), H, W)  # [B,T,2,H,W]
            gt_flow_fl = self._resize_flow_to(self._flow_first_to_last(wan_frames), H, W)  # [B,2,H,W]
            pred_flow_fl = self._resize_flow_to(self._flow_first_to_last(video_btchw), H, W)  # [B,2,H,W]

        gt_01 = ((wan_frames.cpu().float().numpy() + 1.0) / 2.0).clip(0, 1)  # [B,T,3,H,W]
        gen_01 = ((video.cpu().float().numpy() + 1.0) / 2.0).clip(0, 1)  # [B,3,T,H,W]
        first_01 = ((first_frame_bchw.cpu().float().numpy() + 1.0) / 2.0).clip(0, 1)  # [B,3,H,W]
        # Uncropped (with deblur-priming frames) output, for visual
        # inspection of whether that region is actually deblurring the
        # conditioning image as intended -- only meaningful/logged when
        # deblur_prefix_frames > 0.
        gen_with_prefix_01 = (
            ((video_with_prefix.cpu().float().numpy() + 1.0) / 2.0).clip(0, 1)
            if (self.ctx_front + self.ctx_back) > 0 else None
        )  # [B,3,T+front+back,H,W]
        # vjepa_input_clip_01 is already in [0,1] (it's the pre-ImageNet-normalization
        # clip built inside _dynamics_full_grid) -- only set for dynamics_source="vjepa_gt".
        vjepa_input_01 = vjepa_input_clip_01.cpu().float().numpy() if vjepa_input_clip_01 is not None else None

        names = sample.get("name") if isinstance(sample, dict) else None
        metrics_cfg = getattr(self.opt.val, "metrics", None) or {}
        psnr_crop = int(metrics_cfg.get("psnr", {}).get("crop", 8)) if "psnr" in metrics_cfg else 8
        ssim_crop = int(metrics_cfg.get("ssim", {}).get("crop", 8)) if "ssim" in metrics_cfg else 8

        for i in range(bsz):
            idx += 1
            name = names[i] if names is not None else f"{idx:05d}"
            stem = os.path.splitext(os.path.basename(name))[0] + self._val_stem_suffix

            gt_thwc = gt_01[i].transpose(0, 2, 3, 1).astype(np.float32)  # [T,H,W,3]
            gen_thwc = gen_01[i].transpose(1, 2, 3, 0).astype(np.float32)  # [T,H,W,3]

            # Defensively align shapes before stacking/comparing -- gt and
            # gen should already match (same crop, VAE decode reconstructs
            # to the original T/H/W), but this avoids a hard crash on any
            # off-by-one from odd crop_size/vae_scale_factor combinations.
            h = min(gt_thwc.shape[1], gen_thwc.shape[1])
            w = min(gt_thwc.shape[2], gen_thwc.shape[2])
            if self._out_frames_override is not None:
                # resample_time: output length != GT length by design. Keep both whole; for the stacked viz only,
                # resample GT to the output length by nearest index (never used for scoring -- score_speed.py does that).
                n_frames = gen_thwc.shape[0]
                gt_thwc = gt_thwc[:, :h, :w]
                gen_thwc = gen_thwc[:, :h, :w]
                gt_for_viz = gt_thwc[np.linspace(0, gt_thwc.shape[0] - 1, n_frames).round().astype(int)]
            else:
                n_frames = min(gt_thwc.shape[0], gen_thwc.shape[0])
                gt_thwc = gt_thwc[:n_frames, :h, :w]
                gen_thwc = gen_thwc[:n_frames, :h, :w]
                gt_for_viz = gt_thwc
            if self.suite is not None:
                self._clip_stems.append(stem)
                log_video(self.opt, self.accelerator, gt_thwc, f"gt_{stem}", self.global_step)
                log_video(self.opt, self.accelerator, gen_thwc, f"wan_output_{stem}", self.global_step)
                # Common to every experiment (suite and normal alike), on top of the separate clips
                # above: one WAN-INPUT(x burst segments)/VJEPA-INPUT/PRED/GT(/DONOR-GT) stacked video.
                wan_panels = self._wan_input_panels(sample, i, first_01[i])
                vjepa_in_thwc = vjepa_input_01[i].transpose(0, 2, 3, 1).astype(np.float32) if vjepa_input_01 is not None else None
                donor_gt_thwc = None
                if self._donor_viz is not None and self._donor_viz.get("gt01") is not None:
                    donor_gt_thwc = self._donor_viz["gt01"][0].permute(0, 2, 3, 1).cpu().float().numpy()
                stacked_io = self._stacked_input_output_viz(wan_panels, vjepa_in_thwc, gen_thwc, gt_for_viz, donor_gt_thwc)
                log_video(self.opt, self.accelerator, stacked_io, f"stacked_{stem}", self.global_step)
                continue

            # Computed here (rather than at their original spot further
            # below) so they're available for the 2x2 diagnostic grid next;
            # the later logging of these same videos under their own names
            # just reuses these values instead of recomputing them.
            gt_flow_thwc = self._flow_to_video_thwc(gt_flows[i:i + 1])[:n_frames, :h, :w]
            pred_flow_thwc = self._flow_to_video_thwc(pred_flows[i:i + 1])[:n_frames, :h, :w]
            # V-JEPA dynamics-token PCA visualization (top-3 principal
            # components -> RGB, see _vjepa_pca_video_thwc's docstring).
            vjepa_pca_thwc = self._vjepa_pca_video_thwc(dynamics_grid[i], n_frames, h, w)

            # 2x2 diagnostic grid: pred (top-left) / V-JEPA dynamics-token
            # PCA (top-right) over gt (bottom-left) / gt optical flow
            # (bottom-right).
            top_row = np.concatenate([gen_thwc, vjepa_pca_thwc], axis=2)  # [T,h,2w,3]
            bottom_row = np.concatenate([gt_thwc, gt_flow_thwc], axis=2)  # [T,h,2w,3]
            stacked_thwc = np.concatenate([top_row, bottom_row], axis=1)  # [T,2h,2w,3]

            # (1) input video for V-JEPA (the exact clip it saw, Stage A only).
            if vjepa_input_01 is not None:
                vjepa_in_thwc = vjepa_input_01[i].transpose(0, 2, 3, 1).astype(np.float32)  # [T,H,W,3]
                log_video(self.opt, self.accelerator, vjepa_in_thwc, f"vjepa_input_{stem}", self.global_step)
            # (2) input image(s) to Wan (the I2V first-frame condition) -- ALL burst segments
            # stacked side by side when burst_size > 1, not just the first/accessible one.
            wan_panels = self._wan_input_panels(sample, i, first_01[i])
            wan_input_stack_hwc = np.concatenate([img for _, img in wan_panels], axis=1)  # [H, S*W, 3]
            wan_input_stack_chw = wan_input_stack_hwc.transpose(2, 0, 1).astype(np.float32)[np.newaxis]  # [1,3,H,S*W]
            log_image(self.opt, self.accelerator, wan_input_stack_chw, f"wan_input_image_{stem}", self.global_step)
            # ground-truth clip, kept for visual reference alongside the PSNR/SSIM below.
            log_video(self.opt, self.accelerator, gt_thwc, f"gt_{stem}", self.global_step)
            # (3) output video generated by Wan, logged on its own.
            log_video(self.opt, self.accelerator, gen_thwc, f"wan_output_{stem}", self.global_step)
            # (3b) uncropped output including the deblur-priming frames, to
            # visually verify that region is doing what it's meant to.
            if gen_with_prefix_01 is not None:
                gen_with_prefix_thwc = gen_with_prefix_01[i].transpose(1, 2, 3, 0).astype(np.float32)  # [T',H,W,3]
                log_video(
                    self.opt, self.accelerator, gen_with_prefix_thwc,
                    f"wan_output_with_deblur_prefix_{stem}", self.global_step,
                )
            # (4) 2x2 diagnostic grid: pred/vjepa-PCA (top) over gt/gt-flow
            # (bottom) -- quick visual diff plus the two auxiliary panels.
            log_video(self.opt, self.accelerator, stacked_thwc, f"wan_pred_vs_gt_{stem}", self.global_step)

            # (4b) Common to every experiment (suite and normal alike), on top of the separate clips
            # above: one WAN-INPUT(x burst segments)/VJEPA-INPUT/PRED/GT(/DONOR-GT) stacked video
            # (wan_panels already built above for the wan_input_image_{stem} preview -- reused here).
            vjepa_in_thwc = vjepa_input_01[i].transpose(0, 2, 3, 1).astype(np.float32) if vjepa_input_01 is not None else None
            donor_gt_thwc = None
            if self._donor_viz is not None and self._donor_viz.get("gt01") is not None:
                donor_gt_thwc = self._donor_viz["gt01"][0].permute(0, 2, 3, 1).cpu().float().numpy()
            stacked_io = self._stacked_input_output_viz(wan_panels, vjepa_in_thwc, gen_thwc, gt_thwc, donor_gt_thwc)
            log_video(self.opt, self.accelerator, stacked_io, f"stacked_{stem}", self.global_step)

            # (5) optical flow, computed separately for GT and pred: a per-frame
            # flow-visualization video each (gt_flow_thwc/pred_flow_thwc already
            # computed above, reused here), plus a single image with both flow
            # fields drawn as arrows over the actual blurred input (GT in green,
            # pred in red) using the first-to-last-frame displacement -- the
            # field that best matches what physically produced the blur.
            log_video(self.opt, self.accelerator, gt_flow_thwc, f"gt_flow_{stem}", self.global_step)
            log_video(self.opt, self.accelerator, pred_flow_thwc, f"wan_output_flow_{stem}", self.global_step)

            blur_hwc = first_01[i].transpose(1, 2, 0)  # [H,W,3] float in [0,1]
            arrows = self._draw_flow_arrows(blur_hwc, gt_flow_fl[i].cpu().numpy(), color=(0, 255, 0))
            arrows = self._draw_flow_arrows(arrows.astype(np.float32) / 255.0, pred_flow_fl[i].cpu().numpy(), color=(255, 0, 0))
            arrows_chw01 = arrows.astype(np.float32).transpose(2, 0, 1) / 255.0
            log_image(self.opt, self.accelerator, arrows_chw01[np.newaxis], f"flow_arrows_{stem}", self.global_step)

            psnrs, ssims = [], []
            for f in range(n_frames):
                psnrs.append(calculate_psnr(gt_thwc[f], gen_thwc[f], crop=psnr_crop))
                ssim_val, _ = calculate_ssim(gt_thwc[f], gen_thwc[f], crop=ssim_crop)
                ssims.append(ssim_val)
            # Tag by originating source when the val set mixes several
            # (MixedDataset sets sample["source"]). Untagged, every sample in a
            # mixed validation pass logs under the SAME metric name at the SAME
            # step, so a tracker keeping one value per step shows whichever
            # source happens to be logged last -- collapsing two deliberately
            # different probes (does it retain the old capability vs does it
            # learn the new one) into one meaningless number. The two probes
            # also sit at very different absolute levels here, so the collapsed
            # value drifts with sample ordering rather than with model quality.
            metrics = {"video_psnr": float(np.mean(psnrs)), "video_ssim": float(np.mean(ssims))}
            # forward motion error: mean end-point error between RAFT flows of the generated and GT clips (same frame pairs)
            if self.suite is None:
                epe_i = float(torch.linalg.vector_norm(pred_flows[i].float() - gt_flows[i].float(), dim=1).mean())
                metrics["video_epe"] = epe_i
            acc = getattr(self, "_val_acc", None)
            if acc is not None:
                acc["psnr"].append(metrics["video_psnr"]); acc["ssim"].append(metrics["video_ssim"])
                if "video_epe" in metrics:
                    acc["epe"].append(metrics["video_epe"])
            src = self.sample.get("source") if isinstance(self.sample, dict) else None
            if src is not None:
                # collate turns the per-sample string into a list
                tag = src[0] if isinstance(src, (list, tuple)) else src
                metrics = {f"{k}/{tag}": v for k, v in metrics.items()}
            log_metric(self.accelerator, metrics, self.global_step)

        return idx

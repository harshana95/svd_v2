# V-JEPA dynamics -> Wan 2.2 video generation: staged design + build notes

Status: 2026-09-04. Implemented, syntax/unit-tested against a synthetic tiny `WanTransformer3DModel`;
not yet launched against the real 5B checkpoint (no network/GPU in the build environment).

## Goal

Generate a short clip from (a) a first frame and (b) V-JEPA "blur dynamics" tokens, injected into
Wan 2.2 TI2V-5B's self-attention via a trainable adapter -- the Phase-2 follow-on to
`models/reFlux/BlurJepaDynamics_model.py`'s Phase-1 scaffold (blur image -> predicted V-JEPA
token grid, via `BlurJepaViT_arch`; never trained to convergence, only smoke-tested).

## Staging decision (why this isn't built directly on Phase-1)

Rather than wiring the never-yet-trained `BlurJepaViT_arch` predictor straight into Wan, the new
`Wan22VJEPADynamics_model` (`models/MotionBlur2Video/Wan22VJEPADynamics_model.py`) takes a
`dynamics_source` config switch:

- `vjepa_gt` (**Stage A**, default): dynamics tokens = real frozen V-JEPA output computed
  directly on the ground-truth clip itself. An intentional oracle/upper-bound signal, used only
  to validate that the Wan-injection *mechanism* actually steers generation. No dependency on
  Phase-1 -- launchable today.
- `blur_jepa` (**Stage B**): dynamics tokens = `BlurJepaViT_arch`'s prediction from the single
  blurred frame alone (frozen, loaded via `dynamics_encoder_checkpoint`). The deployable model.

Every line downstream of dynamics-token extraction is shared between the two stages. Plan:
launch `checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageA.yml` and its
`_stageA_ablation.yml` twin (`disable_dynamics_injection: true`); only proceed to Stage B once
Stage A clearly beats the ablation on per-frame video PSNR/SSIM, and only once
`checkpoints/reFlux/BlurJepaDynamics.yml` has a real (non-smoke) trained checkpoint.

## Architecture

- **New package** `models/MotionBlur2Video/wan_dynamics_adapters/`: a PGA-style (Physically-
  Guided-Adaptation) gated low-rank value-residual on Wan's self-attention (`WanAttention.attn1`),
  ported from `models/reFlux/blur_adapters/` (the one existing precedent for "inject V-JEPA
  tokens into a frozen diffusion transformer's attention," used by `ContextFluxBlurMap_model.py`
  for Flux2). `JEPAPerceptualEncoder`/`TimestepLayerModulation` are reused unmodified from
  `blur_adapters` (both fully generic). `WanPGAAttnProcessor`/`WanBlockWrapper` are new, adapted
  to Wan's single-stream self-attention (simpler than Flux2's dual noise/cond-token layout).
- **Bug caught and fixed in the ported design**: the *existing* `PGAModule.value_residual` in
  `blur_adapters/pga.py` never actually consumes `jepa_tokens` -- only the (out-of-scope-here)
  `PCMModule`/`BlurResidualCoupler` do. A literal port would have made the new dynamics tokens
  dead code. `WanPGAModule.value_residual` computes its low-rank residual directly from the
  dynamics tokens instead of from the video's own hidden state, while keeping the same
  zero-init/gated structure (`B_v` zero-init + `TimestepLayerModulation.head` zero-init jointly
  guarantee an exact no-op at init -- verified in `test_wan_pga.py`, see below).
- **Token resampling** (`wan_dynamics_adapters/features.py`): V-JEPA's spatial grid (e.g. 24x24)
  has no genuine correspondence to Wan's own latent grid at a given crop size (unlike Flux2/
  `ContextFluxBlurMap_model`, where the *same* image is patchified into both streams) -- so only
  V-JEPA's temporal (tubelet) axis is kept and resampled to Wan's post-patch frame count; the
  spatial axis is globally average-pooled away, and the resulting per-frame vector is broadcast
  uniformly across that frame's spatial tokens.
- **First-frame conditioning**: Wan 2.2 TI2V-5B's own "expand_timesteps" diffusion-forcing
  mechanism, verified directly against `diffusers/pipelines/wan/pipeline_wan_i2v.py`
  (`diffusers==0.39.0.dev0`) rather than assumed -- TI2V-5B has `image_dim=None` (no CLIP-vision
  cross-attention path, unlike the Wan2.1-I2V-14B model `Blur2Vid_Wan_model.py` already trains in
  this repo) and no channel-concat mask. The clean VAE-encoded first frame is broadcast into
  latent position 0 regardless of sampled flow time `t`; the per-token timestep tensor (shape
  `[B, seq_len]`, one value per post-patch token, confirmed via `pipeline_wan_i2v.py`'s own
  `temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()` construction) carries
  timestep=0 at that frame's tokens.
- **Masked flow-matching loss is required, not optional**: frame-0's latent never mixes with
  `noise` regardless of `t`, so `velocity_target = noise - x0` there is an arbitrary,
  unobservable target -- leaving it unmasked injects pure-variance gradient with no learnable
  signal. Verified with a synthetic-transformer test that the masked loss is exactly invariant to
  frame-0's prediction and sensitive to every other frame's.

## What's verified vs. what still needs a real launch to confirm

Verified via unit tests against a tiny synthetic `WanTransformer3DModel` (not the real 5B
checkpoint -- no network access in the build environment):
- PGA injection is an exact no-op at zero-init (`max diff == 0.0`) and does change output once
  perturbed; gradients correctly reach `dynamics_connector`/`modulation.head`/`pga.A_v`/`pga.B_v`.
- The zero-init gate correctly blocks gradient to `modulation`'s upstream layers
  (`t_proj`/`layer_embed`) until `head` itself moves off zero -- expected behavior of the
  zero-init-gate pattern already used (identically) in `blur_adapters`, not a bug.
- Diffusion-forcing tensor mechanics: frame-0 hardwired to the clean condition regardless of `t`;
  per-token timestep is exactly 0 at frame-0's tokens; masked loss ignores frame-0 and is
  sensitive elsewhere; gradient reaches the connector once PGA is off zero-init.
- Gradient checkpointing does *not* break gradient flow to closure-captured adapter parameters in
  this diffusers version, because `ModelMixin`'s `_gradient_checkpointing_func` defaults to
  `use_reentrant=False` (confirmed in `diffusers/models/modeling_utils.py`) -- the classic PEFT
  "`enable_input_require_grads`" gotcha does not apply here. Tested with and without manually
  forcing `hidden_states.requires_grad_(True)`; both paths correctly backprop into the connector.
- `dataset/adobe240_dataset.py`'s new `include_wan_frames`/`wan_num_frames`/`wan_key` opts were
  smoke-tested against the real on-disk Adobe240 data at `/depot/chan129/data/Adobe240/...` --
  correct shapes and value ranges (`wan_frames`: `[9,3,128,128]` in `[-1,1]`, `blur`: `[-1,1]`,
  `frames`: `[0,1]`).

A real bug was caught and fixed during this build: the top-level `vjepa_num_tubelets: 5` default
(copied from Phase-1's config, calibrated for `vjepa_window_size=10`) does *not* match Stage A's
actual V-JEPA input length (`wan_num_frames=9` -> 8 even frames -> **4** tubelets, not 5). Fixed
by deriving `vjepa_num_tubelets` authoritatively at `__init__` time instead of trusting a static
default: empirically, via a real dummy V-JEPA forward pass for Stage A (mirrors
`BlurJepaDynamics_model.__init__`'s own verification convention), and from the loaded
`BlurJepaViT_arch`'s own `.config` for Stage B (which is guaranteed correct by construction,
eliminating any possible config drift against whatever `BlurJepaDynamics.yml` actually trained).

**Not verifiable without a real launch** (flagged, not guessed): the exact `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
checkpoint's `in_channels`/`z_dim`/`scale_factor_spatial` values (code reads these dynamically
from the loaded config everywhere, never hardcoded, so this should be safe regardless of the
exact numbers -- but the very first real run is the actual confirmation). Run the smoke config
first:

```
python trainer.py -opt checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageA_smoke.yml -test
```

## Files

- `dataset/adobe240_dataset.py` (extended, additive-only)
- `models/MotionBlur2Video/wan_dynamics_adapters/` (new package: `context.py`, `features.py`,
  `wan_pga.py`, `wan_wrappers.py`)
- `models/MotionBlur2Video/Wan22VJEPADynamics_model.py` (new model)
- `checkpoints/MotionBlur2Video/Wan22VJEPADynamics_{stageA,stageA_ablation,stageA_smoke,stageB}.yml`

## Validation logging (comet + disk)

`validate_step` logs three artifacts per sample, to both disk (`{experiments_root}/videos/`,
`{experiments_root}/images/`) and comet_ml (via `utils.misc.log_video`/`log_image`, which already
fan out to whatever trackers `report_to` configures):
- `vjepa_input_{name}`: the exact (pre-ImageNet-normalization) clip handed to V-JEPA, as a video
  -- only emitted for `dynamics_source="vjepa_gt"` (Stage A), since Stage B never runs V-JEPA on
  video at all. Exposed by `_dynamics_full_grid` returning `(grid, vjepa_input_clip_01)` instead
  of just the token grid.
- `wan_input_image_{name}`: the single first frame fed to Wan's I2V conditioning.
- `gt_{name}` / `wan_output_{name}`: the ground-truth clip and Wan's generated clip (unchanged
  from the original implementation, renamed from `gen_` to `wan_output_` for clarity).

Bug fixed while wiring this up: none of the four configs defined `val.save_images`, and
`utils/misc.py::log_image` does `if args.val.save_images:` with no default -- the very first
`log_image` call (already present for the input-frame logging) would have crashed with
`ConfigAttributeError` on the first validation pass. Added `save_images: true` to all four
`val:` blocks.

## Stage A: real run against the actual 5B checkpoint -- two experiment directories, easy to conflate

Stage A actually did complete the full 20,000 steps against the real
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` checkpoint, confirming everything flagged above as "not
verifiable without a real launch." The training history spans **two** experiment directories,
which is a real footgun worth documenting explicitly since I (Claude) got this wrong twice in one
session before catching it:

- `Wan22VJEPADynamics_stageA_2026_09_04_21_00_31/` -- the original launch, ran to step 11000
  (`checkpoint-11000`, timestamped 22:56 on 2026-09-04), then was relaunched at a bumped
  `batch_size: 4` with `resume_from_checkpoint: latest` pointed at this same directory.
- `Wan22VJEPADynamics_stageA_2026_09_04_23_00_31/` -- **this** is where that resumed run's
  checkpoints and videos actually landed, because `trainer.py` always creates a fresh
  `{name}/{experiment_key}_{timestamp}/` directory on launch (per CLAUDE.md's documented
  convention), *even when* `resume_from_path` points at a different, older directory. This run's
  latest checkpoint is `checkpoint-20000` (the real final checkpoint), and its `videos/` directory
  has the full `0000k` through `0020k` validation-cycle history for all 8 held-out scenes.

**Practical implication:** any script or comparison that wants "Stage A's final checkpoint" must
use `checkpoint-20000` from the `23_00_31` directory, not `checkpoint-11000` from the `21_00_31`
directory -- the latter is just a mid-training snapshot from before the batch-size bump, not a
completed run. First-attempt mistake, corrected same day: I initially wrote a "correction" here
claiming the run only ever reached 11000 steps and never completed -- that was itself wrong,
based on checking only the `21_00_31` directory. Always check for a *later*-timestamped sibling
experiment directory before concluding a run "stopped early."

## Standalone inference run on checkpoint-20000 (2026-09-05)

Per this project's standing rule (see `infer_and_metrics_on_baseline_completion` memory), rather
than relying only on the training-time validation-cycle videos above, running a real standalone
inference pass via `checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageA_infer.yml`
(`python trainer.py -opt ... -infer`, following the `_infer.yml` pattern already established by
`NAFNet_deblur_mior_infer.yml`/`Restormer_deblur_mior_infer.yml`: `is_train: false`,
`resume_from_checkpoint: checkpoint-20000` explicit rather than `latest`, pointed at the
`23_00_31` directory; `train.max_train_steps: 1` + `validation_steps: 1` to force exactly one
validation pass; `val.max_val_steps: 8` covering all 8 held-out test scenes). Results to be copied
to `/depot/chan129/users/harshana/Blur2Vid/oracle_vjepa_stageA/` alongside the Tedla et al. 2025
baseline for direct comparison (see `tedla2025_blur2vid_outsidephotos/` there) once it completes.

## Ablation gradient-flow bug (found + fixed, 2026-09-05)

Launching the real (non-smoke) `Wan22VJEPADynamics_stageA_ablation.yml` run crashed immediately at
step 1 with `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.
Root cause: both Stage A and its ablation set `lora_finetune: false`, so the transformer itself is
always frozen (`self.transformer.requires_grad_(False)` unconditionally in `__init__`) -- the
*only* trainable parameters in either run are `dynamics_connector`/`modulation`/`pga`
(`_trainable_parameters()`). The original ablation implementation set
`ctx.enabled = not self.disable_dynamics_injection`, i.e. `False`, which made
`WanPGAAttnProcessor.__call__` skip adding the value residual entirely (`wan_pga.py:88-90`) --
so for the ablation, none of the three trainable modules ever entered the graph leading to the
loss, and `loss.backward()` had nothing to differentiate.

Fixed to be a genuinely capacity-matched ablation instead of a no-training placebo: in both
`optimize_parameters` and `validate_step`, when `disable_dynamics_injection` is set, the V-JEPA
dynamics grid is zeroed (`torch.zeros_like`) *before* `_build_dynamics_tokens` rather than
skipping the injection path, and `ctx.enabled` is now unconditionally `True`. The connector/PGA/
modulation modules still run and still train -- they just can only ever learn a content-
independent (constant-per-timestep) residual, since the input carries no scene information. This
isolates "does the *content* of the dynamics tokens help" while holding trainable capacity,
architecture, and injection mechanism fixed between the two arms -- the correct comparison for the
paper's claim, rather than "trained-vs-untrained-network," which the crash-free naive
`ctx.enabled=False` version would have silently been if it hadn't crashed.

## Framework bug found: `utils/misc.py::scandir` chokes on dotfiles inside a scanned symlink

Unrelated to this model's own code, but discovered while debugging the ablation crash (see
`Blur2VidBaseline_findings.md`-equivalent notes below): `models/__init__.py`'s auto-discovery
recursively scans all of `models/` for `*_model.py`. Once `models/Baselines/Blur2Vid/` (a symlink-
containing vendored baseline, see next section) existed, this crashed the *ablation trainer
launch itself* with `NotADirectoryError` on a `.gitignore` file inside a HuggingFace cache
directory reached through that symlink. `_scandir`'s original logic
(`if not entry.name.startswith('.') and entry.is_file(): ... else: recurse`) treated *any*
dotfile-named entry as a directory to recurse into, regardless of whether it actually was one.
Fixed in `utils/misc.py` to skip all dotfiles/dot-dirs outright and only recurse into confirmed
directories (`entry.is_dir()`). This is a latent bug that could resurface anywhere a `.git*`,
`.cache`, or other dotfile ever ends up nested under `models/`, `models/archs/`, or `dataset/`.

## Metrics script verified end-to-end on real data (`analysis/compute_video_metrics.py`)

Ran the new bidirectional patch-PSNR/SSIM/LPIPS + RAFT-EPE + FVD script (built for the baseline-
comparison work, see the paper-comparison plan) against all 88 real `gt_*`/`wan_output_*` video
pairs saved by Stage A's completed run (8 held-out scenes x 11-12 checkpoints each). PSNR/SSIM/
LPIPS/EPE all computed correctly; per-scene PSNR generally improves over training steps within
each scene (e.g. `IMG_0183`: 19.8dB at step 0 to 27dB by step 11k; `IMG_0015` reaches 31-32dB with
LPIPS as low as 0.007). FVD failed on the first attempt: `cdfvd`'s VideoMAE backend (matching
`blur2vid_lib/extra/moMets-*.py`'s own choice) downloads its checkpoint from
`pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com`, which is unreachable from this cluster (and from
the machine running Claude Code) -- confirmed via `curl` timing out from both. Switched to
`cdfvd`'s I3D backend instead (`cdfvd`'s own default; checkpoint hosted on GitHub, confirmed
reachable) -- FVD numbers from the two backends are not directly comparable, so any FVD reported
against Tedla et al.'s own numbers should note this deviation.

Also caught: an earlier CPU-mode smoke-test attempt left a duplicate process running (a botched
`ssh ... "..." &` backgrounding attempt didn't actually terminate locally when expected), and a
second, intentional CPU run was launched without noticing -- both ran concurrently for ~25 minutes
burning all 32 cores on the same node that was also running the ablation trainer's dataloader
workers, before being caught and killed. Always check `ps aux | grep <script>` for duplicates
before assuming a "no output yet" background job is simply slow.

## Blur2Vid (Tedla et al.) baseline: first real inference run (2026-09-05)

`models/Baselines/Blur2Vid/blur2vid_lib/inference.py` (their "outsidephotos" CogVideoX-2b
checkpoint) run against 3 of our own Adobe240 test-split blur images (`IMG_0183/00017.png`,
`IMG_0015/00017.png`, `GOPR9635/00017.png` -- the first two chosen to match scenes already in our
own Stage A validation videos, for direct comparison). Needed two missing-dependency fixes in the
isolated `blur2vid` conda env: `tiktoken` and `protobuf` (both silent `transformers` tokenizer-
loading fallback failures, not the "AS IS" library's own fault). All 3 videos generated
successfully (~62s/video on an idle A100, 50 DDIM steps) to
`/scratch/gilbreth/wweligam/experiments/blur2vid_baseline/eval_outputs/`. Scaled up to all 8 of
Stage A's held-out val scenes shortly after.

## Resolution-consistent comparison: Stage A at 1280x736 vs. blur2vid at 1280x720 (2026-09-05)

The first Stage-A-vs-blur2vid comparison (see the metrics-script section above) evaluated Stage A
at its native 320x320 training crop against blur2vid's native 1280x720 -- a large, uncontrolled
resolution gap that (as later confirmed) meaningfully distorted at least one metric. Re-ran Stage
A's standalone inference at 1280x736 instead (`Wan22VJEPADynamics_stageA_infer_hires.yml`,
`Adobe240Dataset`'s new `resize_mode: resize` option -- a whole-frame, non-aspect-preserving
resize with no cropping, added specifically for this, matching blur2vid's own
`inference.py::convert_to_batch` resize convention exactly). Two real bugs were caught and fixed
getting this to work, both previously latent because every prior resolution used here (320x320
crop, 1280x720 blur2vid) happened not to trigger them:

1. **`mask_strided` shape bug** in both `optimize_parameters` and `validate_step`: the per-token
   timestep mask was built via `first_frame_mask[0, 0, ::p_t, ::p_h, ::p_w]` (Python strided
   slicing, which uses *ceil* semantics for an odd dimension) instead of matching
   `post_patch_height = h_lat // p_h` (*floor* semantics, matching the transformer's own
   `patch_embedding` conv). These silently diverge whenever `h_lat`/`w_lat` is odd -- impossible
   at 320x320 (h_lat always even there) but true at 720/16=45. Fixed by building `mask_strided`
   directly at the already-computed post-patch shape (`torch.ones(post_patch_num_frames,
   post_patch_height, post_patch_width)`, zeroing frame 0) rather than via slicing -- correct
   regardless of parity, since the mask is spatially uniform within any given temporal frame.
2. **720 isn't a valid VAE-latent height at all**, independent of bug 1: Wan's `patch_embedding`
   is a stride==kernel Conv3d with no padding, so `h_lat`/`w_lat` must be evenly divisible by
   `vae_scale_factor_spatial(16) * patch_size(2) = 32` for patchify->unpatchify to round-trip
   losslessly. 720/32=22.5 -- the conv silently drops the last latent row, and the transformer's
   real unpatchified output ends up 1 latent-pixel shorter than every other tensor in the pipeline
   that assumed `h_lat = H // 16` via arithmetic, crashing in `scheduler.step()`. Fixed by using
   736 (nearest multiple of 32 at or above 720) instead of exactly 720; 1280 already satisfies the
   constraint (1280/32=40) so width was unaffected. This means Stage A's comparison resolution is
   1280x736, not pixel-identical to blur2vid's 1280x720 -- close, and both "high resolution" in the
   same sense the mismatched 320x320-vs-1280x720 comparison wasn't, but not bit-for-bit equal.
3. A **third bug, in the metrics script itself** (`analysis/compute_video_metrics.py`), surfaced
   only once evaluating at 736: `_patch_grid`'s `range(0, h, patch)` leaves a small trailing
   partial patch whenever `patch` (40px) doesn't evenly divide the resolution (736/40=18.4) --
   320 and 720 both happen to divide evenly by 40, so this never fired before. A 16px-tall trailing
   patch crashes LPIPS's AlexNet backbone (a pooling layer's output collapses to size 0). Fixed by
   shifting the last patch back to stay full-sized (allowing a little overlap) instead of leaving
   an undersized remainder.

Both the model was *not* retrained at 1280x736 -- still `checkpoint-20000`, trained only at
320x320 -- so this is an out-of-training-distribution resolution test for Stage A, not a
guaranteed-safe one. Results held up reasonably well regardless.

**Resolution-consistent results (n=8, same 8 scenes for both methods):**

| Method | Resolution | PSNR | SSIM | LPIPS | EPE | FVD(i3d) |
|---|---|---|---|---|---|---|
| Stage A (checkpoint-20000) | 1280x736 | 31.33 | 0.8179 | 0.0327 | 5.435 | 285.53 |
| Tedla et al. 2025 (blur2vid) | 1280x720 | 28.40 | 0.7961 | 0.0234 | 5.034 | 1357.17 |

Compare against the earlier, resolution-mismatched numbers (Stage A at 320x320: PSNR=29.34
SSIM=0.8157 LPIPS=0.0334 EPE=2.075 FVD=194.07): PSNR/SSIM/FVD conclusions are essentially
unchanged (Stage A still leads on all three, FVD by a wide margin). **EPE is the one metric that
meaningfully changed**: Stage A's apparent large EPE advantage (2.04 vs. blur2vid's 5.03) mostly
evaporates once evaluated at a comparable resolution (5.44 vs. 5.03, now roughly tied) -- RAFT's
flow estimate at 320x320 is coarser and less sensitive to fine motion error than at ~720p, so the
earlier "Stage A crushes blur2vid on motion accuracy" conclusion was substantially an artifact of
the resolution mismatch, not a real advantage. This is exactly the kind of thing a same-resolution
comparison is supposed to catch, and is worth remembering before drawing conclusions from any
future baseline comparison run at mismatched resolutions.

Results copied to `/depot/chan129/users/harshana/Blur2Vid/oracle_vjepa_stageA/`: the native-320x320
run moved under `native_320x320/` (kept for reference, no longer used for cross-baseline
comparison), the new resolution-consistent run under `hires_1280x736_consistent/` (now the
canonical comparison point against `../tedla2025_blur2vid_outsidephotos/`).

## The ablation vs. Stage A result: Stage A does NOT clearly win (2026-09-05)

`Wan22VJEPADynamics_stageA_ablation.yml` completed its full 20,000-step training run
(`Wan22VJEPADynamics_stageA_ablation_2026_09_05_12_51_21/checkpoint-20000`) on `gilbreth-k004`
while the resolution-consistency work above was happening on the same GPU (~29GB used out of
80GB the whole time, no contention). Evaluated its final checkpoint the exact same way as Stage A
above: standalone inference at 1280x736
(`Wan22VJEPADynamics_stageA_ablation_infer_hires.yml`, `disable_dynamics_injection: true`,
otherwise identical setup), same 8 held-out scenes, same ground-truth videos.

**This is the actual comparison the whole ablation exists for** ("only proceed to Stage B if
Stage A clearly beats this ablation," per every ablation config's own header comment). The result:

| Method | PSNR | SSIM | LPIPS | EPE | FVD(i3d) |
|---|---|---|---|---|---|
| Stage A (real V-JEPA) | 31.33 | 0.8179 | 0.0327 | 5.435 | 285.53 |
| Ablation (content-blind) | 30.59 | 0.8066 | 0.0316 | 5.004 | 262.17 |
| Tedla et al. 2025 (blur2vid) | 28.40 | 0.7961 | 0.0234 | 5.034 | 1357.17 |

**Stage A does not clearly beat the ablation.** The gap (Stage A: +0.74dB PSNR, +0.011 SSIM,
but +0.001 worse LPIPS, +0.43 worse EPE, +23 worse FVD) is small and mixed-direction -- not the
clear win the project's own stated gate for Stage B was looking for, and well within what n=8
scenes could produce as noise. Both Stage A and the ablation clearly beat blur2vid on
PSNR/SSIM/FVD (blur2vid wins LPIPS against both) -- but that comparison doesn't bear on whether
the *V-JEPA content specifically* is doing anything useful, since the ablation gets those same
wins with zero real scene information reaching the model.

**Read this as an open finding, not a final verdict** -- concrete things worth checking before
concluding the injection mechanism doesn't work, in rough priority order:
1. Whether `TimestepLayerModulation`'s `alpha_pga`/`beta_pga` gates in Stage A saturated near
   zero too (i.e. Stage A's own training may have learned to *ignore* its real dynamics tokens,
   converging to something functionally similar to the ablation regardless of what's fed in).
2. Qualitative side-by-side inspection of the actual generated videos (aggregate metrics on n=8
   can hide real per-scene differences a human would immediately notice, or fail to hide
   differences that don't actually look meaningfully different).
3. Whether `pool_dynamics_to_frame_tokens`'s aggressive spatial pooling (global average over
   V-JEPA's full 24x24 grid, discarding all spatial structure and keeping only 5-to-9 frame-level
   vectors) is destroying most of the useful signal before it ever reaches the connector -- this
   was a known design tradeoff (see "Token resampling" in the Architecture section above), not a
   bug, but this result is exactly the kind of evidence that tradeoff might be too aggressive.

Full results, per-scene breakdown, and this table: `/depot/chan129/users/harshana/Blur2Vid/ablation_content_blind/metrics.txt`.

## The spatial-preserving connector redesign broke old-checkpoint resume/eval -- caught mid-investigation (2026-09-06)

Built and trained `deblur_prefix_frames` (prepend N copies of the real clip's own first/sharp
frame before VAE-encoding, giving the model a dedicated extra latent frame whose target is always
"the true sharp first frame" -- see the model docstrings and code comments for the full mechanism)
to test the hypothesis that a frozen/lightly-adapted first latent frame doing double duty
("deblur the input" + "predict subsequent motion") was limiting Stage A. Resumed the completed
LoRA run's checkpoint-20000 for 20,000 more steps with `deblur_prefix_frames: 4` enabled
(`Wan22VJEPADynamics_stageA_lora_deblur.yml`, checkpoint-40000).

**First evaluation was invalid, not just disappointing.** Shortly after this training completed, a
separate redesign (`JEPAFactoredConnector`/`resample_dynamics_grid`, see the section below) replaced
the model's *default* connector construction. Evaluating checkpoint-40000 afterward tried to load a
`JEPAFactoredConnector` module from a checkpoint that only ever saved a `JEPAPerceptualEncoder`
(checkpoints are keyed by class name) -- the loader found no match and silently left the connector at
its random initialization, no error raised. The resulting numbers (PSNR=28.24, SSIM=0.7568,
LPIPS=0.0633, EPE=6.844, FVD=526.66, "a clear regression on every metric") looked like a plausible
disappointing result and were reported and archived as such -- until a follow-up native-320x320 eval
of the *same* checkpoint produced visually psychedelic noise (not just "blurry"), which is what
actually revealed the bug: a real trained model does not degrade to random-looking output at its own
training resolution. See "Module swap checkpoint compat" in memory for the general lesson (always
verify save/load class-name matches, and do a real visual/qualitative check, not just a metrics
read, whenever a result looks like a plausible regression).

**Fix**: added `dynamics_connector_type: "factored" | "pooled"` (default "factored" for fresh runs),
kept the original pooling pipeline in the codebase rather than deleting it, and pinned
`dynamics_connector_type: pooled` explicitly on every existing config that resumes a checkpoint
trained before "factored" existed (Stage A, the ablation, the pre-deblur LoRA run, and this
deblur-prefix run).

**Corrected result, real trained connector confirmed loading (`JEPAPerceptualEncoder_1` in the log,
not `JEPAFactoredConnector_1`)**:

| Method | Step | PSNR | SSIM | LPIPS | EPE | FVD |
|---|---|---|---|---|---|---|
| LoRA (pre-deblur, final) | 20000/20000 | 31.51 | 0.8288 | 0.0314 | 4.477 | 224.21 |
| LoRA + deblur_prefix (final, corrected) | 40000/40000 | 31.56 | 0.8318 | 0.0261 | 2.904 | 203.84 |

**`deblur_prefix_frames` helps** -- better on every metric. PSNR/SSIM are close to a tie (within
likely n=8 noise), but LPIPS (~17% better), EPE (~35% better), and FVD (~9% better) are meaningful
improvements. All 8 scenes also scored in the "forward" temporal direction this time (vs. 6/8
"reversed" in the invalid run), consistent with genuinely correct temporal coherence rather than a
direction-ambiguity artifact. This supports the original motivating hypothesis: giving the model a
dedicated latent frame for deblurring measurably improves subsequent motion quality (EPE, the
flow-based motion-accuracy metric, improved the most).

Full results, per-scene breakdown, and the full incident writeup:
`/depot/chan129/users/harshana/Blur2Vid/lora_deblur_final/metrics.txt`.

## Spatial-preserving connector: final result does not beat deblur_prefix alone (2026-09-07)

Built `resample_dynamics_grid` + `JEPAFactoredConnector` (see the "Architecture" section's Token
resampling notes for the motivation -- the original connector global-average-pools V-JEPA's 24x24
spatial grid into one vector per frame before projecting via a single dense
`Linear(1024, wan_inner_dim=3072)`; the new pipeline keeps the spatial grid, bilinearly resizes it
to Wan's own post-patch H,W, and projects via two small heads (per-head-dim 128, per-attention-head
24) combined by outer product instead of one dense matrix). Trained fresh (LoRA + deblur_prefix +
this new connector, `Wan22VJEPADynamics_stageA_lora_deblur_spatial.yml`, 20,000 steps -- could not
resume any prior checkpoint since the connector's class changed, see the section above) to
completion (checkpoint-20000), evaluated the same way as everything else here:

| Method | Step | PSNR | SSIM | LPIPS | EPE | FVD |
|---|---|---|---|---|---|---|
| LoRA + deblur_prefix (pooled connector) | 40000/40000 | **31.56** | **0.8318** | **0.0261** | **2.904** | **203.84** |
| LoRA + deblur + spatial (interim, 14000/20000) | 14000/20000 | 31.01 | 0.8231 | 0.0301 | 3.474 | 237.77 |
| LoRA + deblur + spatial (final) | 20000/20000 | 31.37 | 0.8204 | 0.0299 | 3.559 | 209.01 |

**Does not beat deblur_prefix alone on any metric** -- slightly worse on all five, though the gaps
are small (not a dramatic regression like the connector-bug incident above). It did improve from
interim to final on most metrics (hadn't fully converged at 14000 steps), and still clearly beats
Stage A/Ablation/blur2vid, landing roughly on par with plain LoRA (no deblur). This does *not*
necessarily mean preserving V-JEPA's spatial structure doesn't matter -- unconfounded factors worth
separating in any follow-up: (a) `JEPAFactoredConnector` has far fewer parameters than the dense
one (1024*(128+24)=155,648 vs 1024*3072=3,145,728), a capacity difference independent of whether
spatial structure is preserved; (b) V-JEPA's 24x24 grid, already a fairly coarse tokenization, may
not carry much spatially-localized signal to begin with; (c) single run, not repeated for noise.

## Overall recipe progression (summary, as of 2026-09-07)

| Step | Change | Result |
|---|---|---|
| 1 | Frozen backbone, real V-JEPA (Stage A) vs. content-blind control (Ablation) | Inconclusive -- Stage A did not clearly beat the ablation (see above) |
| 2 | + LoRA-unlocked backbone | Clear win over both frozen variants |
| 3 | + `deblur_prefix_frames=4` (dedicated latent frame for deblurring) | Clear further win (confirmed only after fixing a connector-checkpoint-loading bug, see above) -- **current best** |
| 4 | + spatial-preserving V-JEPA connector (`JEPAFactoredConnector`) | No further win -- roughly neutral, slightly behind step 3 |
| 5 | + `aspect_ratio_range: [0.5, 2.0]` (batch-consistent variable-aspect-ratio training crop, pooled connector) | Clear further win at the non-square 1280x736 eval resolution -- **new current best** |

**Current best recipe**: LoRA + `deblur_prefix_frames=4` + `aspect_ratio_range: [0.5, 2.0]`, pooled
(dense) connector, checkpoint-60000 at
`/scratch/gilbreth/wweligam/experiments/Motion/Wan22VJEPADynamics_stageA_lora_deblur_ar_2026_09_07_03_38_44/`
(load with `dynamics_connector_type: pooled`). All comparison results, videos, and per-scene metrics
for every step above are under `/depot/chan129/users/harshana/Blur2Vid/`.

### Step 5 detail: aspect_ratio_range (2026-09-07)

`Adobe240Dataset` gained an `aspect_ratio_range` option (batch-consistent variable crop aspect
ratio, area held ~constant, via a new `collate_fn`) to help the model generalize to
non-320x320-square resolutions at test time. Resumed checkpoint-40000 for 20,000 more steps
(`Wan22VJEPADynamics_stageA_lora_deblur_ar.yml`) to checkpoint-60000, then evaluated the same way
as every other row (1280x736, same 8 held-out scenes):

| Method | Step | PSNR | SSIM | LPIPS | EPE | FVD |
|---|---|---|---|---|---|---|
| LoRA + deblur_prefix (pooled) | 40000/40000 | 31.56 | 0.8318 | 0.0261 | 2.904 | 203.84 |
| LoRA + deblur + aspect_ratio_range | 60000/60000 | **32.09** | **0.8510** | 0.0269 | **2.623** | **163.37** |

Wins on 4 of 5 metrics (PSNR +0.53, SSIM +0.019, EPE -0.28, FVD -40.5); LPIPS is essentially tied
(+0.0008). Consistent with training on varied aspect ratios actually helping generalization to a
non-square eval resolution, rather than relying purely on RoPE's resolution-flexibility from
square-only training. Full results at `/depot/chan129/users/harshana/Blur2Vid/lora_deblur_ar_final/`.

**Methodology bug caught and fixed while computing this row**: the first metrics attempt ran
`compute_video_metrics.py` directly against the raw `experiments_root/videos/` folder using its
default `gt_`/`wan_output_` prefixes, and silently picked up the `gt_flow_*`/`wan_output_flow_*`
optical-flow *visualization* videos (logged by `Wan22VJEPADynamics_model.validate_step` alongside
the real content) as if they were additional real GT/pred pairs -- `find_pairs()`'s naive prefix
globbing doesn't distinguish `gt_SCENE.mp4` from `gt_flow_SCENE.mp4`. This corrupted the result
(n=16 instead of 8, EPE inflated to 14.4 by garbage flow-vs-flow scores instead of the real 2.6).
Every other row in this doc was already computed against a manually curated staging folder
(`gt_<scene>.mp4`/`pred_<scene>.mp4` only, under `/scratch/gilbreth/wweligam/experiments/blur2vid_baseline/`)
which never had this problem -- this run's first attempt just skipped that curation step. Fixed by
curating the same kind of clean staging folder before re-running the metrics script. Any future
metrics run against a raw `videos/` folder should do the same, or `compute_video_metrics.py`'s
`find_pairs` should be hardened to exclude `flow_`-prefixed stems.

## Qualitative-only check on MIOR and GoPro (2026-09-07)

Ran checkpoint-40000 (current best recipe) on 8 images each from Deblur_MIOR (`val/blur`) and
GoPro (`test/GOPR0384_11_00/blur`) -- both datasets provide only single blur/gt(-or-sharp) image
pairs, no high-fps burst, so there's no real motion ground truth to score against. Added a new
`dataset/blur_single_image_dataset.py::BlurSingleImageDataset` for this (loads a blur image, plus
a `wan_frames` placeholder that's just the blur image repeated `wan_num_frames` times so the
model's existing T/H/W-shape-derivation code path runs unmodified). Ran with
`disable_dynamics_injection: true`, since PGA's value-residual formula gives `dv=0` exactly for
zero input regardless of the checkpoint's learned weights -- feeding zeros is a mathematically
clean no-injection pass, unlike feeding V-JEPA features extracted from a static repeated image
(which would be extracted from non-existent motion and mean nothing). This is purely a look at
what the trained backbone + LoRA + deblur-prefix mechanism alone produces from a single
out-of-domain blurred photo, not a scored comparison.

Configs: `Wan22VJEPADynamics_bestckpt_infer_{MIOR,GoPro}.yml`. Results (videos + input images) at
`/depot/chan129/users/harshana/Blur2Vid/{mior_qualitative,gopro_qualitative}/` -- `wan_output_*.mp4`
is the useful output; the logged `gt_*`/`wan_pred_vs_gt_*`/PSNR/SSIM are not meaningful (compared
against the repeated-blur-image placeholder, not real motion).

## Stage B real training launched, redesigned architecture (2026-09-07)

`checkpoints/MotionBlur2Video/BlurJepaViT_stageB.yml`: the first real (non-smoke) training run of
the blur -> V-JEPA-token predictor, previously only ever smoke-tested (see
`checkpoints/reFlux/BlurJepaDynamics.yml`/`_smoke.yml`). Two things changed going into this run:

- **`BlurJepaViT_arch` redesigned**: replaced pool-the-blur-image-then-broadcast-across-T (a single
  static spatial map, tiled identically across all 5 timesteps with only an additive per-timestep
  bias) with a fixed set of `T*grid_h*grid_w` learned query embeddings -- one per (timestep, height,
  width) output position, genuinely distinct parameters -- that cross-attend into the blur image's
  own spatial encoding through `CrossAttentionBlock`s (self-attn among queries + cross-attn into the
  blur context + MLP). Standalone shape/backward test: 38.6M params, correct fixed `[B,2880,1024]`
  output for both square and non-square inputs, gradients reach `patch_embed`.
- **`dataset/adobe240_patchifier_dataset.py::Adobe240PatchifierDataset`** (new): amortizes disk IO
  by loading `batch_size` full bursts once and drawing `patches_per_group` different
  aspect-ratio-varying crops from that same in-memory group before reloading, same producer pattern
  as `FullResPatchifierDataset` elsewhere in this repo. Batch size raised 8 -> 32,
  `aspect_ratio_range: [0.5, 2.0]` added to the train crop. Requires the dataset's own
  `gt_key`/`lq_key` mirrored as top-level attributes (models read `dataloader.dataset.gt_key`
  directly) and `return_name: false` on the train split (Accelerate's IterableDataset dispatch path
  chokes concatenating a mixed str+tensor dict -- both caught and fixed during smoke-testing).

## Stage C: chaining Stage B into Stage A for real blur-to-video generation (2026-09-07)

`checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageC_{,_hires}.yml`: the actually-deployable
pipeline. Pure inference, no training -- combines Stage A's best trained Wan weights
(checkpoint-60000) with Stage B's `BlurJepaViT_arch` checkpoint via `Wan22VJEPADynamics_model`'s
existing `dynamics_source: blur_jepa` switch (designed from the start, never exercised until Stage
B had a real checkpoint). Evaluated on Adobe240's test split (real GT exists there, unlike
MIOR/GoPro) at both native 320x320 and the 1280x736 hires resolution, against Stage B's interim
checkpoints (4000-5000/20000): `video_psnr` roughly 21-29dB, `video_ssim` roughly 0.70-0.94 at
both resolutions -- clearly below Stage A's oracle numbers but not degenerate, and the hires result
generalized about as well as native despite `BlurJepaViT_arch` never being trained above ~320-scale
crops (its fixed-query-count design doesn't hard-code an input resolution).

**Bug found from watching the actual Stage C output videos**: in most cases the generated video
plays backwards relative to true motion, and in some cases *part* of a scene runs one direction
while the rest runs the other. Root cause: a single blurred image is fundamentally time-symmetric
-- a blur trail from A to B looks identical to one from B to A, so there is no way to recover
*which end came first* from the blurred image alone. Training `BlurJepaViT_arch` with plain MSE
against one specific (arbitrary, given the ambiguity) temporal order gives the model no consistent
signal to learn a "forward" convention, and -- because the redesigned architecture lets each of
the 2880 (t,h,w) query positions attend to the blur image independently, with only a self-attention
pass to coordinate them -- different spatial regions can each independently settle on a different
guessed direction, producing the observed partial-reversal artifact. (Stage B's own token-space
validation metrics, plain MSE/cosine-similarity, didn't surface this -- a scrambled temporal order
can still score reasonably on those coarse whole-grid metrics; it only became visible after
decoding to actual video in Stage C.)

**Fix**: `BlurJepaDynamics_model`'s loss is now flip-invariant, computed as ONE scalar PER SAMPLE
(never per spatial position): `_flip_temporal` reverses the tubelet (T) axis of the target token
grid (tubelet-major reshape, matching `wan_dynamics_adapters/features.py`'s own convention),
`_flip_invariant_mse` takes `min(MSE(pred, target), MSE(pred, target_flipped))` per whole sample and
backprops through whichever is smaller. Computing this as a single global choice (not per-position)
is what should eliminate the partial-reversal failure mode: a prediction that mixes forward content
in one region with reversed content in another scores *worse* under both alignments than a
prediction that commits fully to either direction everywhere, so gradient descent is pushed toward
whole-clip consistency, not per-position guessing. The model may still sometimes play a whole clip
backwards relative to the *original* GT -- that specific ambiguity is genuinely unrecoverable from
a single blurred image and isn't a bug -- but every position within one generated clip should now
agree on which direction that is. `validate_step` also now logs `flip_rate` (fraction of validation
samples where the reversed alignment scored better) as an ongoing diagnostic.

Sanity-tested standalone (flip is a correct involution, correctly identifies pre-flipped samples
with zero loss) and via the existing smoke config before relaunching. The interim run described
above (checkpoints up to step ~6300/20000) was trained *without* this fix and was killed; Stage B
was restarted from scratch at
`/scratch/gilbreth/wweligam/experiments/Motion/BlurJepaViT_stageB_2026_09_07_18_42_56/` with the
corrected loss. Stage C should be re-evaluated against this run's checkpoints once it has trained
enough to compare, both for overall quality and specifically for whether the partial-reversal
artifact is gone.

## Stage B (flip-invariant run) completed; Stage C re-evaluated on the final checkpoint (2026-09-08)

The restarted Stage B run completed its full 20,000-step budget (`checkpoint-20000`, loss settled
~2.04-2.05, comparable to the pre-fix run's plateau -- the fix changes *what* the loss measures,
not the underlying task difficulty). Re-ran Stage C (both native 320x320 and 1280x736 hires) against
this final checkpoint, and -- unlike the earlier interim checks, which only used
`Wan22VJEPADynamics_model.validate_step`'s coarse forward-only PSNR/SSIM (not diagnostic for this
specific fix, since a model that now *consistently* commits to a globally wrong-but-coherent
direction would score similarly there) -- also ran the same bidirectional patch-metrics script used
for every Stage A row, which explicitly tests both temporal alignments per scene:

| Resolution | PSNR | SSIM | LPIPS | EPE | FVD | Directions (fwd/rev out of 8) |
|---|---|---|---|---|---|---|
| native 320x320 | 28.65 | 0.7975 | 0.0246 | 3.033 | 207.69 | 7/1 |
| hires 1280x736 | 30.71 | 0.8101 | 0.0280 | 5.916 | 293.44 | 6/2 |

Both resolutions show every scene landing clearly on ONE global alignment (not a mediocre score
split between both, which a genuinely mixed/inconsistent clip would produce) -- consistent with the
fix working, though this per-scene test only distinguishes "all-forward" vs "all-reversed" and can't
by itself rule out a *partial* mid-clip flip; confirming that fully requires watching the actual
videos. Native EPE (3.033) is close to Stage A's own best hires EPE (2.623, checkpoint-60000) --
genuinely usable quality from a purely blur-driven prediction, not a real burst. Hires generalizes
reasonably (BlurJepaViT_arch was never trained above ~320-scale crops) with somewhat higher
PSNR/SSIM but also higher EPE/FVD -- expected for a resolution well outside its training
distribution, not a red flag.

All results (videos, input images, per-scene bidirectional metrics) copied to depot:
`/depot/chan129/users/harshana/Blur2Vid/stageC_blur_jepa_final20000_{native,hires}/`.

**Depot reorganization**: every existing Stage A result folder under
`/depot/chan129/users/harshana/Blur2Vid/` was renamed with a `stageA_` prefix (e.g.
`lora_deblur_ar_final` -> `stageA_lora_deblur_ar_final`) for consistency now that Stage B/C results
share the same directory; `tedla2025_blur2vid_outsidephotos` (an external baseline, not one of this
project's own stages) was left as-is.

## Baselines for the Stage C result (2026-09-08)

At the same 1280x736 hires resolution / same 8 scenes:

| Method | PSNR | SSIM | LPIPS | EPE | FVD |
|---|---|---|---|---|---|
| Tedla et al. 2025 (blur2vid, external baseline) | 28.40 | 0.7961 | 0.0234 | 5.034 | 1357.17 |
| Content-blind ablation (our architecture, zero motion signal) | 30.59 | 0.8066 | 0.0316 | 5.004 | 262.17 |
| Stage A oracle (real V-JEPA, upper bound) | 32.09 | 0.8510 | 0.0269 | 2.623 | 163.37 |
| Stage C (deployable, blur image only) | 30.71 | 0.8101 | 0.0280 | 5.916 | 293.44 |

Stage C clearly beats the external baseline (Tedla et al.) on PSNR/SSIM/FVD, with FVD in
particular showing a large gap (293 vs. 1357). Against our own content-blind ablation, Stage C
wins on PSNR/SSIM/LPIPS but is worse on EPE and FVD -- worth being honest about: Stage B's
predicted-but-imperfect motion tokens currently produce slightly more accurate per-pixel content
but somewhat less coherent *motion* than just letting the backbone generate its own default motion
with zero steering. Real signal, clearly better than the external baseline, but not yet
unambiguously better than "no guidance at all" on every axis.

## Stage B, second predictor architecture: LoRA-finetune V-JEPA itself instead of training from scratch (2026-09-08)

`checkpoints/MotionBlur2Video/BlurJepaViT_stageB_vjepalora.yml`: `BlurJepaDynamics_model` gained a
`predictor_type: "from_scratch" | "vjepa_lora"` switch. Motivation: the from-scratch predictor
(`BlurJepaViT_arch`, 38.6M params, random init) converges slowly -- loss plateaued around 2.04-2.05
after the full 20,000-step budget, still not obviously converged.

`vjepa_lora`: loads a SECOND V-JEPA instance (same pretrained weights as the frozen target),
LoRA-adapts it via the same `apply_lora_to_model` helper already used for Wan's LoRA in this
project. Confirmed by inspection that V-JEPA is a standard timm-style ViT (`blocks.N.attn.qkv`
fused projection + `blocks.N.attn.proj`, NOT diffusers-style separate q/k/v/o), so
`target_modules=["qkv", "proj"]` targets exactly the 48 attention-related Linear layers (24 blocks
x 2), leaving the 48 MLP layers frozen. The blurred image is fed as `vjepa_window_size` IDENTICAL
repeated copies (a fake "clip"), so V-JEPA's own tubelet_size=2 grouping produces the identical
`[B,2880,1024]` token-grid shape as the real target -- same task as the from-scratch predictor
(extract whatever motion signal is spatially encoded in the blur trail), but starting from V-JEPA's
own pretrained visual representations instead of random weights. Kept in fp32 throughout (not
bf16) -- V-JEPA's hub weights use fp32 attention internally, same caveat as every other V-JEPA
caller in this repo. Same flip-invariant, per-sample loss as the from-scratch run.

Only 1.18M/305.86M params trainable (0.39%). Smoke test result was striking: loss dropped from
1.79 to 0.848 within 5 steps, vs. the from-scratch predictor's ~3.5-4.2 starting loss -- strong
evidence that starting from V-JEPA's pretrained weights gives a dramatically better initialization.
Real training run launched at
`/scratch/gilbreth/wweligam/experiments/Motion/BlurJepaViT_stageB_vjepalora_2026_09_08_12_28_46/`,
20,000-step budget (same as the from-scratch run, for a fair comparison) -- given the much faster
initial convergence, it may plateau well before that; worth checking validation loss regularly
rather than assuming the full budget is needed.

`Wan22VJEPADynamics_model.py`'s `blur_jepa` loading code gained a matching
`dynamics_predictor_type: "from_scratch" | "vjepa_lora"` switch so Stage C can consume this
predictor type: loads a fresh V-JEPA base + `peft.PeftModel.from_pretrained(base, checkpoint_path)`
onto it (pointed at the saved `.../checkpoint-N/PeftModel_1/` adapter directory, not the
`BlurJepaViT_arch_1/` HF directory the from-scratch predictor saves), then feeds the blur image
through the same repeated-frame "fake clip" trick as `BlurJepaDynamics_model._predict_tokens`'s
vjepa_lora branch.

## Investigated: "higher cosine similarity but worse video" -- ruled out a loading bug, found a real (unresolved) disconnect (2026-09-08)

First Stage C test against this predictor (checkpoint-2000/20000, ~10% through) gave
`video_psnr` (19.4, 27.6) / `video_ssim` (0.576, 0.939) -- not clearly better than the from-scratch
predictor's own interim result, despite training-time logs showing much higher cosine similarity
(~0.9) than the from-scratch predictor ever reached (~0.6). Given this exact "trained metric looks
great, checkpoint doesn't actually work" pattern already burned this project once before (see the
PEFT-resume-optimizer-disconnect incident in memory), this was investigated rigorously rather than
assumed to be noise:

1. **Save/load round-trip verified byte-exact**: loaded a fresh V-JEPA base +
   `PeftModel.from_pretrained` onto checkpoint-4000's saved adapter, then compared all 96 LoRA
   tensors (48 A + 48 B matrices) against the raw `adapter_model.safetensors` file key-by-key
   (matching through PEFT's `.default.` adapter-name segment in the in-memory state dict, which
   isn't present in the saved file's keys) -- 0 mismatches. The save/load mechanism is exact.
2. **The adapter has a real, correctly-directioned effect**: on a real validation sample, the
   loaded predictor scores cosine=0.854/mse=1.08 against the true target, clearly better than the
   *same* frozen V-JEPA base fed the *same* repeated-blur input with **no** LoRA adapter at all
   (cosine=0.776/mse=1.71). If loading had silently failed, these would be indistinguishable --
   they aren't.
3. **From-scratch's own checkpoints are genuinely training too** (sum(abs diff) between
   checkpoint-19000 and 20000 = 5794, clearly nonzero -- not a frozen/no-op encoder).
4. **Direct, same-sample, same-computation comparison**: vjepa_lora checkpoint-4000 (20% through)
   scores cosine=0.854/mse=1.08 vs. from-scratch checkpoint-20000 (FULLY CONVERGED)
   scoring cosine=0.662/mse=2.19 -- confirms the user's "0.9 vs 0.6" observation is real, and shows
   it holds even against from-scratch's *best* checkpoint, not just an early one.

Given checkpoint-4000 was never actually tested in Stage C (the first test used the earlier,
less-converged checkpoint-2000), re-ran Stage C against checkpoint-4000 specifically to see if the
confirmed token-level win would show up in video quality. **It didn't**: `video_psnr` (19.97, 27.89)
/ `video_ssim` (0.602, 0.943) -- essentially unchanged from checkpoint-2000's result, despite a
large, verified token-level improvement in between. So the "wrong checkpoint" theory was only
partially right (checkpoint-2000 genuinely was less-converged and worth retesting), but does not
fully explain the disconnect -- **there appears to be a real gap between aggregate token-space
cosine similarity/MSE (computed by flattening the entire [1,2880,1024] grid into one vector) and
downstream video quality after going through Wan's frozen dynamics-injection pipeline**, at least
for the vjepa_lora predictor type. No further loading/saving bug found; this looks like a genuine,
unresolved ML question rather than a code defect. Plausible (untested) explanations worth
investigating later: (a) a global flattened-vector cosine/MSE can hide per-tubelet or per-channel
structural errors that matter more to `pool_dynamics_to_frame_tokens`'s temporal resampling than to
the aggregate metric; (b) vjepa_lora's errors, being perturbations of the *same* V-JEPA manifold
the target lives on, might be "confidently wrong" in a way more disruptive to the frozen
`dynamics_connector`/PGA pathway (trained only on real oracle V-JEPA statistics) than
from-scratch's possibly more "regressed-to-the-mean" error pattern; (c) letting both predictors
train further (from-scratch is already at its full budget; vjepa_lora has plenty of budget left)
before drawing final conclusions.

### Root cause found: the naive comet metric can't see past the direction ambiguity (2026-09-08)

The above investigation ruled out a loading/saving bug and confirmed the tokens are genuinely
better at every representation stage checked (raw grid, spatially-pooled, and post-connector --
see below), but that still left "why doesn't better tokens mean better video" unexplained. Pushed
further:

- **Pooled representation (what `dynamics_connector_type: pooled` actually uses downstream --
  spatial detail is averaged away, only a per-tubelet mean survives)**: still clearly favors
  vjepa_lora (cosine 0.983 vs 0.881 mean over 6 samples). Rules out "vjepa_lora's advantage is in
  spatial detail the pipeline throws away anyway."
- **Post-connector representation (after the real trained `JEPAPerceptualEncoder`, i.e. one step
  from what actually reaches Wan)**: still clearly favors vjepa_lora (cosine 0.939 vs 0.624, MSE
  0.30 vs 1.63, norm 188.0 vs 165.3 -- target norm 194.9, so vjepa_lora's magnitude is also closer).
  Rules out the whole token/connector pipeline as the site of the problem.
- **The actual mechanism**: ran the *bidirectional* patch-metrics script (the same one used for
  every Stage A row, which scores both the forward and time-reversed alignment and reports
  whichever is better) on the vjepa_lora checkpoint-4000 Stage C videos -- previously this was
  never done for these interim checks, only the training-time `validate_step`'s naive forward-only
  comet `video_psnr`/`video_ssim`. Result: **7 of 8 scenes score "reversed" as the better
  alignment** -- a far higher reversal rate than any other checkpoint in this project (Stage A
  checkpoints typically land 6-7/8 *forward*). Once scored correctly:

  | Method | PSNR | SSIM | LPIPS | EPE | FVD |
  |---|---|---|---|---|---|
  | from-scratch (native, checkpoint-20000, fully converged, bidirectional) | 28.65 | 0.7975 | 0.0246 | 3.033 | 207.69 |
  | vjepa_lora (native, checkpoint-4000, 20% through, bidirectional) | **29.41** | **0.8189** | **0.0242** | **2.196** | **203.95** |

  vjepa_lora's checkpoint-4000 beats the from-scratch predictor's *fully converged* result on
  every metric, using a fifth of the training budget -- fully consistent with its clear token-level
  advantage. The earlier "disconnect" was a metric-choice artifact, not a real quality gap: the
  video content genuinely is better, it's just frequently generated in the reversed temporal
  direction, which a direction-blind metric can't distinguish from actually-bad content. This is
  not a bug to fix (the flip-invariant loss deliberately makes either direction acceptable, and
  direction is genuinely unrecoverable from a single blurred image) -- it does mean
  **vjepa_lora's checkpoints should always be scored with the bidirectional metric going forward**,
  never the naive training-time comet number, and any future Stage C check (native or hires,
  interim or final) should run `analysis/compute_video_metrics.py` before drawing conclusions.
  Open, not yet investigated: *why* vjepa_lora reverses so much more often than from-scratch (7/8
  vs. typically 1-2/8) -- plausibly some interaction between V-JEPA's own pretrained temporal
  priors and the repeated-identical-frame input trick, but unconfirmed.

## Remaining candidate causes for the residual Stage C (vjepa_lora) vs. Stage A gap (2026-09-08)

The section above resolved the *headline* disconnect (a direction-blind metric, not a quality
gap). What it did **not** settle is the follow-on question: if the predicted tokens score
cosine~0.9 (0.983 pooled, 0.939 post-connector) against the Stage B target, why isn't Stage C
simply *equal* to Stage A? Ranked candidates, cheapest-to-settle first. Nothing here reopens the
save/load round-trip, the pooled-vs-spatial-detail question, or the connector pipeline -- all
three are already empirically ruled out above.

**1. The comparison itself is not matched -- on resolution OR on Stage A checkpoint (most likely
the largest single factor).** Stage C vjepa_lora's bidirectional row (29.41 / 0.8189 / 0.0242 /
EPE 2.196 / FVD 203.95) is at **native 320x320**. Every Stage A number being used as the reference
(32.09 / 0.8510 / EPE 2.623 / FVD 163.37) is at **1280x736 with checkpoint-60000**. The only
resolution-matched Stage A number anywhere in this doc is the native-320 row from the *frozen-
backbone* checkpoint-20000 recipe: 29.34 / 0.8157 / 0.0334 / EPE 2.075 / FVD 194.07. That row is from the *frozen-backbone*
step-1 recipe -- the one that did not beat the content-blind ablation -- whereas Stage C loads
checkpoint-60000, the step-5 recipe that is ~2.5dB better at hires. So Stage C matching it (29.41
vs 29.34 PSNR, 0.8189 vs 0.8157 SSIM, 0.0242 vs 0.0334 LPIPS, close on EPE/FVD) does **not** show
the gap is absent -- Stage C may still be well behind what checkpoint-60000 itself would score at
native 320. What this item establishes is that the comparison currently being used to *define* the
gap is uninformative: it varies resolution and Stage A recipe at the same time, exactly the failure
mode documented in the 2026-09-05 resolution-consistency section, where a mismatched-resolution
comparison inflated an EPE conclusion that evaporated once matched. **Missing measurements**: (a) Stage A checkpoint-60000
(the exact Wan weights Stage C loads) at native 320, and/or (b) Stage C vjepa_lora at hires
1280x736 with bidirectional scoring. (a) is the single highest-value run on this list --
it is the only one that turns "uninformative comparison" into a real number. Neither has been run.

**2. Even a PERFECT Stage B predictor is out-of-distribution for Stage A's frozen connector:
4 tubelets vs 5, over different source frames.** Stage A's oracle feeds V-JEPA `wan_frames` =
`frames01_full[1:10]` (9 frames), of which `even_t = (9//2)*2 = 8` are used -> **4 tubelets**
(2304 tokens) -- this is already documented as a caught bug in the "What's verified" section above
(`Wan22VJEPADynamics_model.py:318` derives it empirically and overrides the config's
`vjepa_num_tubelets: 5`). Stage B's regression target is V-JEPA on `frames01_full[0:10]` (10
frames) -> **5 tubelets** (2880 tokens), a different clip length, different temporal position
embeddings, and a one-frame window offset. So `pool_dynamics_to_frame_tokens`'s `F.interpolate` resamples
from 5 source points where it was trained on 4 -- certain, and a real resampling-phase mismatch on
its own. The stronger claim (that the frozen `JEPAPerceptualEncoder` is also fed a genuinely
different *feature distribution*, not just a different tubelet count) depends on how V-JEPA's
temporal position embeddings behave at clip length 8 vs 10, and is **not yet verified**. One
forward pass settles it: run V-JEPA on the same 8 frames at native length 8 vs. padded/extended to
10, and compare the pooled per-tubelet vectors. Close => this item reduces to the resampling-phase
issue (real, smaller); far apart => it is as strong as stated. No shape check fires, because the only assertion is that N factors as
`tubelets*grid_h*grid_w`. This affects both predictor types equally (both target the 5-tubelet
grid), which is consistent with *both* Stage C variants sitting below Stage A. Fix would be to
retrain Stage B against an 8-frame/4-tubelet target (set `vjepa_window_size: 8` and confirm
`wan_start` alignment), or to retrain Stage A's connector on 5-tubelet input.

**3. The 7/8 reversal rate carries a real, unavoidable score penalty that Stage A never pays --
even under bidirectional scoring.** `analysis/compute_video_metrics.py:95` reverses the whole
prediction (`pred_rev = pred_thwc[::-1]`) and compares it against the un-reversed GT. But Wan's
diffusion-forcing conditioning hardwires the clean GT **first** frame into latent position 0, so
pred frame 0 is essentially pixel-perfect by construction. Under a "reversed" alignment that
guaranteed-perfect frame is scored against GT's *last* frame -- the single best frame in the clip
becomes one of the worst. The magnitude is unverified -- pred frame 0 is VAE-round-tripped and scored
by patch-PSNR at patch size 40, not compared as raw pixels, so treat "one near-perfect frame
becoming one of the worst, over ~9 frames" as an order-of-magnitude illustration, not a number.
Stage A lands forward 6-7/8 and almost never pays this; vjepa_lora pays it
7/8. This is a genuine cost of the flip-invariant loss's freedom, not a scoring bug, and it
plausibly accounts for a large share of whatever residual gap survives item 1. Testable directly:
recompute the bidirectional metrics excluding frame 0, or report the forward/reversed per-frame
PSNR curve.

**4. Every cosine reported so far includes the temporal-mean (DC) component, which the injection
pathway largely cannot use.** `JEPAPerceptualEncoder` is a single `nn.Linear`
(`models/reFlux/blur_adapters/jepa_connector.py:14`) and `WanPGAModule.value_residual` is
`alpha*(df@A_v@B_v) + beta*df` (`wan_dynamics_adapters/wan_pga.py:45`) -- the whole path is affine
in the dynamics tokens, and with the `pooled` connector the per-frame vector is broadcast
identically across every spatial token. A component constant across all N tokens is therefore just
a constant bias added to `value`, largely inert. **The only part that actually steers generation is
the temporal deviation `d = pooled_t - mean_t(pooled)`.** The 0.983 pooled / 0.939 post-connector
cosines were computed on the raw vectors, mean included; the deviation subspace has never been
measured separately. (For reference: cosine 0.939 with a norm ratio 188.0/194.9 already implies
~34% relative L2 error on the raw vector; on the mean-removed part it could be far larger.)
**Decisive cheap test, no Wan required, one V-JEPA forward per sample**: for ~8 val samples compute
`d` for (a) the oracle on the real clip, (b) from_scratch, (c) vjepa_lora, then measure cosine of
`d` *between different samples* for each predictor. If vjepa_lora's `d` is near-identical across
samples while the oracle's is not, its temporal variation is a content-independent artifact of
V-JEPA's temporal position embeddings acting on the repeated-identical-frame input (LoRA r=8 on
`qkv`/`proj` only, with pos-embeds and all MLPs frozen, has limited capacity to change that) --
which would also explain the anomalous 7/8 reversal rate the section above flagged as open.

**5. Stage A's own headroom over "no signal at all" is small, so "similar to Stage A" is a low
bar and n=8 noise is comparable to the effect.** Stage A (31.33) vs. the content-blind ablation
(30.59) at hires is +0.74dB, mixed-direction on the other four metrics. The entire oracle-to-
content-blind range is roughly the size of the differences being interpreted here.

**6. checkpoint-4000 is 20% through a 20,000-step budget** -- the cleanest confound to remove, and
free (the run is already training).

**7. Minor, listed only so they don't get chased.** Stage B trains the predictor under
`mixed_precision: bf16` autocast (the encoder is in `self.models`, so `accelerator.prepare` wraps
it) while Stage C runs it in fp32 outside `self.models` -- numerically small. At hires,
`vjepa_from_unit_tensor` squashes 1280x736 to a 384x384 square, an aspect distortion only partly
covered by Stage B's train-time `aspect_ratio_range` (its val split is square 320).
**Confirmed fine, do not investigate**: `dynamics_vjepa_repeat_frames: 10` matches Stage B's
`vjepa_window_size: 10`; Stage C val `crop_size: 320` matches Stage B's val split; the
tubelet-major reshape convention is shared across `_flip_temporal`,
`pool_dynamics_to_frame_tokens` and `resample_dynamics_grid`.

### Quantifying it: ~74% of the PSNR discrepancy is the resolution mismatch alone (2026-09-08)

The doc already contains a *controlled* resolution experiment -- Stage A checkpoint-20000, same
model, same 8 held-out scenes, native 320x320 vs. 1280x736 (the 2026-09-05 resolution-consistency
section). Its measured native -> hires deltas: **PSNR +1.99**, SSIM +0.0022, LPIPS -0.0007,
EPE +3.36, FVD +91.5. Applying those to Stage C's native row:

| | Stage C native (measured) | + resolution delta | Stage A ckpt-60000 hires | residual |
|---|---|---|---|---|
| PSNR | 29.41 | 31.40 | 32.09 | **-0.69** |
| SSIM | 0.8189 | 0.8211 | 0.8510 | -0.0299 |
| LPIPS | 0.0242 | 0.0235 | 0.0269 | **+0.0034 (Stage C better)** |
| EPE | 2.196 | 5.556 | 2.623 | -2.93 (unreliable, see below) |
| FVD | 203.95 | 295.41 | 163.37 | -132.0 (unreliable, see below) |

So of the 2.68 dB PSNR gap that prompted the question, **1.99 dB (74%) is accounted for by
resolution alone**, leaving ~0.69 dB. LPIPS actually *inverts* -- Stage C is already better.

Two caveats on the correction. (a) It is first-order: the deltas were measured on checkpoint-20000
(frozen backbone, square-only training), while Stage C loads checkpoint-60000, which was trained
*with* `aspect_ratio_range` specifically to improve non-square hires generalization -- its own
resolution response is almost certainly different (ckpt-60000 scores EPE 2.623 at hires where
ckpt-20000 scored 5.435). (b) Consequently the EPE and FVD rows above should not be trusted at all;
those two are exactly the metrics most sensitive to resolution and least stable at n=8. **EPE and
FVD are where a real quality gap most plausibly still hides; PSNR/SSIM/LPIPS is where it mostly
doesn't.**

**Most likely overall cause of the discrepancy, in order:** (1) the resolution mismatch above,
~74% of the PSNR gap; (2) the 7/8 reversal penalty (item 3 -- the guaranteed-perfect
diffusion-forced frame 0 being scored against GT's last frame), which is a systematic per-clip
PSNR/SSIM cost Stage A almost never pays and is comfortably large enough to cover the remaining
~0.69 dB; (3) only then anything token-side. Note that any "the predicted tokens are actually
worse" explanation has to get past post-connector cosine 0.939 at a near-matched norm
(188.0 vs 194.9) -- the one version that survives that is item 4 (the reported cosines include the
temporal-mean/DC component, which the affine connector+PGA path largely cannot use).
Still the same two runs that settle it: **checkpoint-60000 at native 320**, and the mean-removed
deviation test.

### Missing measurement (b) now run: Stage C vjepa_lora at hires, bidirectional (2026-09-08)

Ran it (checkpoint-6000/20000, 30% through Stage B training -- checkpoint-4000, used for the
native row above, had been rotated away by `checkpoints_total_limit=3` by the time this launched,
so this is not the literally-identical checkpoint, only the closest available): PSNR=31.70,
SSIM=0.8376, LPIPS=0.0254, EPE=4.624, FVD=221.89 (7/8 scenes reversed, same pattern as native).

This lands almost exactly where the resolution-delta extrapolation above predicted (31.40 PSNR
predicted vs. **31.70 actual** -- within 0.3 dB, and the small excess is consistent with
checkpoint-6000 being somewhat further trained than the checkpoint-4000 the extrapolation was
based on). **Strong direct confirmation that resolution mismatch, not a real quality gap, was the
dominant cause of the discrepancy**: at matched hires resolution, vjepa_lora (31.70) now sits only
0.39 dB below Stage A's own checkpoint-60000 oracle ceiling (32.09) -- comfortably inside the
7/8-reversal-penalty budget item 3 above already argues should exist. LPIPS is essentially tied to
the oracle at this resolution too (0.0254 vs. 0.0269, Stage C still slightly better). EPE/FVD
remain worse (4.624 vs 2.623 EPE, 222 vs 163 FVD) -- exactly the two metrics item 1(b)'s caveat
already flagged as resolution-unstable and least trustworthy at n=8, so this is not new evidence of
a token-side problem, consistent with the "in order" ranking above.

Missing measurement (a) -- Stage A checkpoint-60000 itself at native 320x320 -- is still not run;
it remains the single highest-value remaining check to fully close out item 1, since it's the only
piece that would let native-vs-native and hires-vs-hires both be compared without extrapolation.

## Stage B (vjepa_lora) completed its full 20,000-step budget; final Stage C numbers (2026-09-09)

`checkpoints/MotionBlur2Video/BlurJepaViT_stageB_vjepalora.yml` finished its full run (loss
~0.66-0.82 at the end). Re-ran Stage C at both native and hires resolution against the true final
checkpoint-20000 (same checkpoint for both this time, unlike the earlier interim checks where
`checkpoints_total_limit=3` had rotated away the checkpoint used for one resolution before the
other could reuse it -- always verify a referenced checkpoint still exists before launching against
it, per the standing lesson in memory). All numbers below via the bidirectional patch-metrics
script.

| Resolution | PSNR | SSIM | LPIPS | EPE | FVD | Directions (fwd/rev) |
|---|---|---|---|---|---|---|
| native 320x320 | 29.39 | 0.8172 | 0.0231 | 2.099 | 202.10 | 2/6 |
| hires 1280x736 | 31.62 | 0.8374 | 0.0256 | 5.061 | 216.00 | 2/6 |

Both are the best results yet for this predictor line (native EPE 2.099 and LPIPS 0.0231 beat
every earlier interim checkpoint checked). Full final comparison, everything at 1280x736 hires,
same 8 scenes:

| Method | Step | PSNR | SSIM | LPIPS | EPE | FVD |
|---|---|---|---|---|---|---|
| Tedla et al. 2025 (external baseline) | -- | 28.40 | 0.7961 | 0.0234 | 5.034 | 1357.17 |
| Content-blind ablation | 20000/20k | 30.59 | 0.8066 | 0.0316 | 5.004 | 262.17 |
| Stage C, from-scratch predictor | 20000/20k | 30.71 | 0.8101 | 0.0280 | 5.916 | 293.44 |
| **Stage C, vjepa_lora predictor (FINAL)** | 20000/20k | **31.62** | **0.8374** | **0.0256** | 5.061 | 216.00 |
| Stage A (oracle, real V-JEPA, ceiling) | 60000 | 32.09 | 0.8510 | 0.0269 | 2.623 | 163.37 |

**Final conclusion for this line of work**: a predictor that sees nothing but a single blurred
image, fully converged, lands within 0.47 dB PSNR and 0.0136 SSIM of Stage A's oracle ceiling
(which uses the real motion burst) and actually beats the oracle on LPIPS. It clearly and
consistently beats the from-scratch predictor, the content-blind ablation, and the external
baseline on every metric except EPE/FVD -- and per the extensive investigation above, those two
are the metrics most distorted by the 7/8 reversal rate and by resolution instability at n=8, not
necessarily evidence of a remaining real quality gap. Results copied to depot:
`/depot/chan129/users/harshana/Blur2Vid/stageC_vjepalora_final20000_{native,hires}/`.

Still open, not yet run: missing measurement (a) from the section above (Stage A checkpoint-60000
at native 320x320, to fully remove the extrapolation from the native-resolution comparison), and
the still-unexplained *why* of vjepa_lora's much higher reversal rate than from-scratch.

## New session goal (2026-09-10): minimize the reversal rate, rule out stationary/generic prediction

User flagged that some Stage C (vjepa_lora) outputs look like still videos, hypothesizing the
LoRA-finetuned predictor may be giving the blurred image's tokens for every temporal tubelet (a
content-independent, effectively-stationary prediction). This directly targets item 4 in the
"remaining candidate causes" section above, which was flagged but never actually tested. Ran the
decisive diagnostic that section proposed, plus one more control the advisor tool recommended.

### Diagnostic: cross-sample cosine of the mean-removed temporal deviation `d`

For 6 Adobe240 test-split samples, computed `d = pooled_t - mean_t(pooled_t)` (the per-tubelet
pooled `[5,1024]` token grid with its own temporal mean removed -- the only part the affine
connector+PGA path can actually use, per item 4) for three sources: the frozen real-burst target,
the checkpoint-20000 vjepa_lora predictor (fed the repeated-blur fake clip), and -- the added
control -- the **frozen, un-adapted target V-JEPA itself fed that same repeated-blur fake clip**
(no LoRA at all; isolates what's architecturally inherent from what training changes).

| Source | `d` cross-sample cosine (mean of 15 pairs) | Sign structure |
|---|---|---|
| Frozen target, real burst | 0.51 | all positive (0.34-0.71) |
| **Frozen target, repeated-blur input (no LoRA)** | **0.79** | **all positive (0.66-0.89)** |
| vjepa_lora predictor, checkpoint-20000 | 0.37 | bimodal: sample 0 negative vs. all others (-0.13 to -0.28), samples 1-5 all positive with each other (0.44-0.86) |

Per-sample temporal std (magnitude of `d`) for the lora predictor is consistently ~15-20% lower
than the real target's, not collapsed to ~0.

**Reading:** the frozen, completely untrained V-JEPA already produces a strong, near-generic
"shape" template (cosine 0.79, no sign structure at all) when fed identical repeated frames -- this
is an architectural artifact of its positional embeddings acting on a temporally-degenerate input,
not something LoRA training causes. LoRA training on top of that *does* do real work: it pulls the
cross-sample cosine down to 0.37, meaningfully below even the real target's own 0.51, and the
magnitude reduction is modest (15-20%), not a collapse. **So the "stationary/generic prediction"
hypothesis as literally stated is not what's happening** -- the predictor is not just re-emitting
the blurred image's tokens for every tubelet; it has learned genuine, mostly content-dependent
temporal structure.

What *is* real: the bimodal sign split. One sample's `d` is anti-correlated with all five others,
which are mutually positively correlated. This is exactly the pattern expected if the predictor
learned essentially one dominant temporal "shape" (inherited, partly, from the frozen template) and
applies it with a per-sample sign/polarity that is only weakly tied to which direction is actually
correct -- because **the flip-invariant training loss (`_flip_invariant_mse`, per-sample
`min(mse_fwd, mse_rev)`) gives literally zero reward for matching the dataset's true forward
orientation specifically.** Picking "reversed" costs nothing as long as it's internally consistent.
Combined with a strong pre-existing generic-shape prior whose default polarity has no reason to
line up with true-forward, this is a coherent, testable explanation for vjepa_lora's 6-7/8 Stage-C
reversal rate vs. 1-2/8 for the from-scratch predictor (which has no such prior to inherit and
mostly follows whatever weak real signal exists in the blur trail).

### Fix implemented and launched (v2 run, 2026-09-10)

Two changes, both in `models/reFlux/BlurJepaDynamics_model.py`, targeting the two contributing
factors above:

1. **More LoRA capacity**, to reduce reliance on the frozen generic template: r 8->16, alpha
   16->32, `predictor_lora_target_modules` adds `mlp.fc1`/`mlp.fc2` on top of the existing
   `attn.qkv`/`attn.proj` (all confirmed-Linear layers in V-JEPA's timm-style ViT blocks) --
   6.29M trainable params vs. ~3.1M before (2.02% of the 311M-param backbone).
2. **New `predictor_direction_bias_weight` option in `_flip_invariant_mse`** (default 0.0, exact
   backward compat). Adds `direction_bias_weight * mse_fwd` on top of the existing
   `min(mse_fwd, mse_rev)`: reversed can still win the primary term when it's a strictly better
   fit (real per-sample ambiguity from a single blurred frame is respected), but every sample now
   also gets a small continuous gradient nudging toward the real forward order, so
   coin-flip/borderline cases drift toward correct instead of locking onto an arbitrary default.
   `is_flipped` itself (used for reporting) is computed from `mse_fwd`/`mse_rev` alone, unaffected
   by the bias term, so reversal-rate numbers stay comparable across configs. Set to 0.1 for this
   run.

New configs: `checkpoints/MotionBlur2Video/BlurJepaViT_stageB_vjepalora_v2{,_smoke}.yml`.
Smoke-tested clean (96 matching Linear layers found for `[qkv,proj,fc1,fc2]`, `loss_all` /
`loss_loss_mse` / `loss_flip_rate` all logged correctly end to end). Real run launched:
`/scratch/gilbreth/wweligam/experiments/Motion/BlurJepaViT_stageB_vjepalora_v2_2026_09_10_14_08_22/`,
same 20,000-step budget as v1. Once complete: re-run Stage C (native + hires, bidirectional
metrics only) and compare the reversal rate directly against v1's 6-7/8, plus re-check PSNR/SSIM/
LPIPS/EPE/FVD against the v1 final table above to confirm the added capacity/regularizer didn't
regress quality while fixing direction.

## v2 fix results: reversal rate meaningfully reduced, every metric improved (2026-09-10)

Stage B v2 completed its full 20,000-step budget (final `loss_all` ~0.70-0.71, `loss_loss_mse`
~0.64-0.69 -- same converged range as v1's 0.66-0.82, slightly better). Verified the saved adapter
config before evaluating: `r=16, alpha=32, target_modules=[fc2, attn.proj, fc1, qkv]` -- matches
intent exactly. Training-time `loss_flip_rate` (a noisy per-batch proxy, not the real signal) rose
from ~0.13 early to a stable ~0.23-0.26 plateau for the back 70% of training and never trended
toward 0 -- on its own this looked inconclusive/mildly discouraging, which is exactly why this
project's standing rule is to never trust a training-time proxy over real Stage C inference.

New configs: `checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageC_vjepalora_v2{,_hires}.yml` --
verified to differ from their v1 siblings in *only* `experiment_key` and
`dynamics_encoder_checkpoint` (diffed line-by-line before launching), so this is a clean,
confound-free comparison: same Stage A checkpoint-60000, same PGA/LoRA/deblur-prefix/connector
settings, same 8 held-out scenes, only the Stage B predictor differs.

Ran both resolutions, bidirectional metrics only (per this doc's standing rule for this
predictor line):

| Resolution | Version | PSNR | SSIM | LPIPS | EPE | FVD | Directions (fwd/rev) |
|---|---|---|---|---|---|---|---|
| native 320x320 | v1 | 29.39 | 0.8172 | 0.0231 | 2.099 | 202.10 | 2/8 |
| native 320x320 | **v2** | **29.44** | **0.8197** | **0.0228** | **2.012** | **189.92** | **3/8** |
| hires 1280x736 | v1 | 31.62 | 0.8374 | 0.0256 | 5.061 | 216.00 | 2/8 |
| hires 1280x736 | **v2** | **31.79** | 0.8415 | 0.0257 | **4.593** | **195.43** | **4/8** |

**Every metric improved or held at both resolutions** (native: all 5 metrics improved; hires: 4 of
5 improved, LPIPS essentially tied at +0.0001) -- the added LoRA capacity and direction-bias
regularizer did not trade quality for anything. **The reversal rate improved at both resolutions**,
and by a much larger margin at hires: native went from 2/8 to 3/8 forward (a modest step), hires
went from 2/8 to **4/8 forward -- an even split**, down from a strong 75% reversal bias to
essentially a coin flip. This is the single largest effect of the fix, and it lands hires exactly
where you'd expect if the residual reversals are now driven by genuine per-sample direction
ambiguity (some blur trails really are close to symmetric) rather than a learned, mostly
content-independent default polarity.

Results copied to depot: `/depot/chan129/users/harshana/Blur2Vid/stageC_vjepalora_v2_final20000_{native,hires}/`
(33 files each, same convention as every other row in this project).

**Read on the two original session hypotheses.** (a) "Stationary/generic blur-frame prediction":
the diagnostic above already showed this wasn't literally happening (v1's predicted temporal
deviation had genuine, mostly content-dependent structure, just ~15-20% lower magnitude than the
real target's) -- v2's capacity increase nonetheless improved every quality metric, consistent with
the LoRA-capacity bottleneck being real even if it wasn't a full collapse. (b) "Time reversal":
substantially, if not completely, mitigated -- 6/8->4/8 at hires. Whether the remaining ~50%
reversal rate is now the genuine, irreducible ambiguity floor (a single blurred frame really can't
disambiguate direction in general) or could be pushed lower still with a larger
`predictor_direction_bias_weight` is the natural next question, but is not pursued further in this
session -- the fix as implemented met the stated goal (minimize reversal, rule out
stationary-prediction as the cause) without any quality trade-off, which is the actionable
stopping point.

## Stage B v2 extended to 100k steps, then stopped at checkpoint-50000 -- and a major correction (2026-09-10/12)

Given v2's training loss and `loss_flip_rate` had both plateaued well before step 20000, launched
`checkpoints/MotionBlur2Video/BlurJepaViT_stageB_vjepalora_v2_100k.yml`: the SAME v2 recipe
(r=16 LoRA, `[qkv,proj,fc1,fc2]`, `predictor_direction_bias_weight=0.1`), resumed in-place from
checkpoint-20000 via `path.use_same_dir` + `resume_from_checkpoint`, extended to a 100,000-step
budget to see whether more optimization (not more capacity or a stronger bias weight) could push
the reversal rate down further.

**Infrastructure history** (relevant for future runs, not the science): the run started on
gilbreth-k000, whose 7-day `sinteractive` walltime expired mid-run (2026-09-11, `TIMEOUT` in
`sacct`) -- not a crash, just the interactive allocation's own time limit. Checkpointing had saved
cleanly up to checkpoint-32000, so nothing of substance was lost. Continued on gilbreth-k039 via
`resume_from_checkpoint: latest` (robust to whatever checkpoint actually survived), sharing that
node's GPU with an unrelated legitimate job for the rest of the run -- iteration time degraded from
~1.35s/it (dedicated) to ~3.0-3.2s/it under contention, pushing the ETA for the full 100k budget out
to 50+ hours.

**Training-time flip_rate never moved.** From step ~20000 through step ~50000 (30,000+ steps,
spanning the k000->k039 continuation), `loss_flip_rate` stayed flat in a ~0.19-0.27 band with no
directional trend at any point, and `loss_loss_mse` stayed in the same ~0.60-0.71 range the whole
time. This was read as a genuine, well-established training plateau -- at the user's request, the
run was **stopped at checkpoint-50000** (halfway through the 100k budget) rather than waiting out
the rest.

### The correction: checkpoint-50000 beats checkpoint-20000 substantially at the real Stage C level

Ran the same bidirectional Stage C evaluation (native + hires, identical Stage A checkpoint-60000,
identical 8 held-out scenes) against checkpoint-50000 that was run against checkpoint-20000
earlier. Two infrastructure notes on getting there: (a) two attempts to check an interim
checkpoint (36000) both OOM'd on GPU VRAM due to unrelated concurrent jobs sharing gilbreth-k039 --
deferred rather than forced; (b) the first checkpoint-50000 hires attempt, on a freshly-suggested
node (gilbreth-k006), was killed by the Linux OOM-killer at ~32.8GB **system RAM** even though the
GPU itself was completely free (0/82GB) -- that allocation had only requested `--mem=32G`
(standby QOS). **Wan2.2 TI2V-5B hires inference needs just over 32GB system RAM during model
loading, a requirement entirely separate from GPU VRAM** -- use `--mem=64G+` for any standby-queue
allocation running this pipeline's hires config. Retried successfully on gilbreth-k038
(`--mem=64G`).

| Resolution | Checkpoint | PSNR | SSIM | LPIPS | EPE | FVD | Reversed |
|---|---|---|---|---|---|---|---|
| native | v1, 20000 | 29.39 | 0.8172 | 0.0231 | 2.099 | 202.10 | 6/8 |
| native | v2, 20000 | 29.44 | 0.8197 | 0.0228 | 2.012 | 189.92 | 5/8 |
| native | **v2, 50000** | **29.46** | **0.8198** | **0.0226** | **1.779** | **144.73** | 4/8 |
| hires | v1, 20000 | 31.62 | 0.8374 | 0.0256 | 5.061 | 216.00 | 6/8 |
| hires | v2, 20000 | 31.79 | 0.8415 | 0.0257 | 4.593 | 195.43 | **4/8** |
| hires | **v2, 50000** | **31.98** | **0.8460** | **0.0250** | **3.310** | **168.80** | 5/8 |
| hires | Stage A oracle (ceiling) | 32.09 | 0.8510 | 0.0269 | 2.623 | 163.37 | -- |

**Every quality metric (PSNR/SSIM/LPIPS/EPE/FVD) improves monotonically across all three
checkpoints at both resolutions.** The gains from 20000->50000 are the largest single jump in this
entire predictor-architecture line of work, especially on EPE and FVD -- exactly the two metrics
this doc's own earlier analysis flagged as "where a real quality gap most plausibly still hides."
The gap to the Stage A oracle ceiling has nearly closed at hires:

| Metric | v1 gap to oracle | v2(20k) gap | v2(50k) gap |
|---|---|---|---|
| PSNR | -0.47 | -0.30 | **-0.11** |
| SSIM | -0.0136 | -0.0095 | **-0.0050** |
| LPIPS | -0.0013 | -0.0012 | **+0.0019 (beats oracle)** |
| EPE | +2.438 | +1.970 | **+0.687 (3.5x smaller)** |
| FVD | +52.63 | +32.06 | **+5.43 (~10x smaller)** |

**The one metric that is NOT monotonic is the hires reversal rate**: 6/8 (v1) -> 4/8 (v2, 20000)
-> 5/8 (v2, 50000) -- worse than checkpoint-20000, though still much better than v1. The native
reversal rate, by contrast, IS monotonic (6/8 -> 5/8 -> 4/8). This means the direction-bias fix and
the extended training are doing somewhat different things: capacity/bias-weight reduced the
generic-template reversal bias (documented above), while continued training mostly improved
motion-content fidelity (EPE/FVD) rather than further fixing direction -- consistent with the
training-time `loss_flip_rate` proxy staying flat (it only measures token-space alignment, which
this second effect doesn't touch) while real video quality kept improving underneath it.

**Central correction for future work on this project**: *the training-time `loss_flip_rate`/loss
plateau does NOT mean real Stage C video quality has stopped improving.* This is the same lesson
("never trust the training-time proxy over real inference") the project already knew from the
naive-metric-vs-bidirectional-metric episode, now proven a second time in the opposite direction --
here the proxy *understated* real improvement rather than overstating quality. **The 100k
extension was very likely stopped prematurely**; the monotonic trend across 20000->50000 gives no
reason to expect the improvement had exhausted itself by step 50000. If this predictor line is
revisited, resuming from checkpoint-50000 to a higher step count (or the original full 100k) is the
most promising next action, not further architecture or loss changes.

Results copied to depot: `/depot/chan129/users/harshana/Blur2Vid/stageC_vjepalora_v2_100k_ckpt50000_{native,hires}/`
(33 files each, same convention as every other row in this project).

## New direction: burst-of-blur-images -> longer video, targeting the reversal problem at its root (2026-09-13)

Everything above treats the reversal problem as something to *mitigate* (LoRA capacity, direction-bias
regularizer) within a fundamentally time-symmetric input: a single blurred image looks identical played
forward or backward, so no amount of training can make the correct direction *recoverable* from it -- only
*guessable*, which is exactly what the 50/50-ish reversal rates above reflect (confirmed via advisor
consultation: this read of the single-frame line's ceiling holds up).

New idea, requested directly: condition on a **temporally-ordered BURST of `burst_size` (starting at 3)
blurred images** instead of one. A burst is *not* time-symmetric as a sequence -- segment i's content
precedes segment i+1's by construction -- so the correct direction should be recoverable from the input
itself, not dependent on a learned prior. This should also let Wan generate a proportionally longer video
(one segment's worth of motion per burst element instead of one short clip total).

### Design (advisor-reviewed before implementation)

- **Dataset** (`dataset/adobe240_dataset.py`, additive `burst_size` param, default 1 = byte-identical to
  before): groups each scene's own listed blur centers into **non-overlapping** chunks of `burst_size`
  consecutive centers (stride=burst_size, so bursts never share a segment), *before* interleaving across
  scenes (grouping after interleaving would collapse a validation batch onto one scene -- caught by
  advisor). Reuses the *existing* precomputed per-center blur files directly (no new blur synthesis) --
  verified spacing between consecutive listed centers is 8 frames, smaller than the 11-frame averaging
  window, so consecutive segments' windows overlap by ~3 frames; harmless (each blur image still mostly
  reflects its own local motion), and advisor confirmed this reuse-existing-files choice over synthesizing
  non-overlapping blur from scratch. Returns `burst_blur` `[B,S,3,H,W]` (pm1) and `burst_frames`
  `[B,S,vjepa_window_size,3,H,W]` (01, per-segment real sub-windows for the oracle), plus `wan_frames`
  (same key as before) now sliced from the *full continuous span* covering all S segments' windows
  end-to-end (`stride*(S-1)+window_size` frames), not any single segment's window.
- **Arithmetic (advisor-reviewed): burst_size=3.** Full span = 8*(3-1)+11 = 27 frames, trimmed centered to
  `wan_num_frames=25=4*6+1` (Wan's `(n-1)%4==0` constraint, only 2 frames of slack trimmed off each end).
  burst_size=5 would give span 43 -> trim to 41, also clean, saved for a later scale-up.
- **Model** (`models/MotionBlur2Video/Wan22VJEPADynamics_model.py`, additive `burst_size` config, default
  1): new `_dynamics_full_grid_burst` runs the *exact same* per-segment dynamics logic burst_size times
  (once per segment) and **concatenates the resulting token grids along the tubelet axis** before handing
  off to the existing `pool_dynamics_to_frame_tokens`/`resample_dynamics_grid`/connector/broadcast/PGA
  pipeline completely unchanged -- `self.vjepa_num_tubelets` is set to the TOTAL
  (`burst_size * per_segment_vjepa_num_tubelets`) at init time, so nothing downstream of the concatenation
  needs to know burst mode exists at all. Verified end-to-end via a startup shape assertion (same pattern
  as the existing single-frame check): `burst_size=3 x per_segment_vjepa_num_tubelets=5 = 15`, `seq_len=80`
  matched Wan's real patchified token count exactly on the first try.
- **Critical correctness point caught by advisor, applied before writing any code**: the Stage-1 oracle
  (`dynamics_source=vjepa_gt`) must run V-JEPA **per-segment-then-concatenate, NOT on the whole burst
  video at once**. Running V-JEPA once on the full 25-frame clip would produce a *different* tubelet count
  and content distribution than what a per-segment Stage-2/3 predictor (which only ever sees one blurred
  image at a time) could ever reproduce -- this project already has this exact failure mode on record
  (the 4-tubelet-oracle-vs-5-tubelet-predictor mismatch, see the "root cause found" entry above). The oracle
  must be architecturally identical in construction to what the deployable pipeline will feed it.
- **Deblur-priming prefix reused unchanged, applied ONCE**: `_first_frame_condition` returns the *first*
  burst segment's blurred image as the I2V condition (not per-segment repeats) -- `deblur_prefix_frames`
  primes the very start of the whole burst, exactly mirroring the single-frame mechanism.
- **Sequencing (advisor-reviewed)**: Stage 1 (oracle) is the only piece that must be trained fresh; the
  existing Stage B v2 checkpoint-50000 predictor is directly, immediately applicable *per-segment*
  (identical window_size=11/vjepa_window_size=10 single-image-to-tokens task) for a first Stage-3
  (deployable, chained) datapoint -- no dedicated Stage 2 burst-specific retrain is needed to get a first
  end-to-end result. Whether a dedicated Stage 2 finetune adds anything is deferred until after that first
  datapoint exists.
- **The one check that discriminates most (next up)**: bidirectional-metrics direction labels on Stage 1
  (oracle) at an early checkpoint, *before* any Stage 2/3 work. The oracle definitely contains the true
  direction; if oracle-burst output still reverses, the pooled-connector/PGA path is destroying temporal
  order and no predictor-side work would fix it. If it comes back solidly forward, the burst hypothesis is
  confirmed and the rest is execution.

### Status

Warm-started Stage 1 config (`Wan22VJEPADynamics_stageA_burst3.yml`, burst_size=3, resumed from the
existing best single-frame recipe's checkpoint-60000 into a NEW experiment folder -- LoRA/connector/PGA
weights are all shape-independent of burst_size, so this is a valid, compute-saving warm start, not a
dependency on the single-frame *predictor* work) smoke-tested cleanly end-to-end
(`Wan22VJEPADynamics_stageA_burst3_smoke.yml`, exit 0, checkpoint saved, validation videos logged) and
launched for real training on gilbreth-k013 (20000 more steps, 60000->80000, matching this project's
established "20000 steps per new mechanism" pattern). Results pending.

## First real result: burst-mode oracle checkpoint-65000, 8/8 forward (2026-09-13)

The critical check (advisor's point #3 above) ran at the very first opportunity -- checkpoint-65000,
only 5000 steps into the 20000-step burst-adaptation budget (warm-started from the single-frame
Stage A's checkpoint-60000) -- via `Wan22VJEPADynamics_stageA_burst3_infer_native.yml` and the usual
bidirectional patch-metrics script:

```
Method                                      PSNR    SSIM   LPIPS     EPE      FVD  n
Stage1 burst3 oracle ckpt65000             27.09  0.7424  0.0265   3.367   191.61  8
```

Per-scene direction (all 8/8 scenes scored better in the FORWARD direction -- none reversed):
720p_240fps_1, GOPR9635, GOPR9637a, IMG_0004a, IMG_0015, IMG_0023, IMG_0179, IMG_0183 -- every single
one forward.

**This is the strongest direction-correctness result anywhere in this project so far.** Every single-
frame Stage C checkpoint (v1, v2(20k), v2(50k)) sat close to 50/50 (2/8 to 5/8 forward) no matter how
much training or loss-shaping was applied, because a single blurred image is fundamentally
time-symmetric -- the advisor-confirmed ceiling of that whole line of work. Feeding a temporally-
ordered burst instead, even at just 5000 steps of adaptation and even still in "oracle" mode (V-JEPA
sees the real burst, not a predicted one), immediately drives the reversal rate to 0/8. This is exactly
the mechanism the burst redesign was meant to test (advisor: "concatenation preserves [segment] order
by construction... a flip-invariant per-segment predictor should still yield globally forward video").

Quality is not yet at the single-frame oracle's ceiling (PSNR=32.09/SSIM=0.8510/LPIPS=0.0269/
EPE=2.623/FVD=163.37) -- expected, since the model has had only 5000 steps to adapt from a 9-frame/
5-tubelet regime to a 25-frame/15-tubelet one (3x more real motion content to reconstruct). Notably,
LPIPS is already essentially at parity (0.0265 vs 0.0269) and FVD/EPE, while behind, are not
dramatically so. Training continues toward the full 80000-step budget; the expectation (informed by
the checkpoint-20000->50000 experience above, where downstream video quality kept improving well past
where the training-time loss proxy looked flat) is that PSNR/SSIM/EPE/FVD will continue closing this
gap with more steps, while the reversal rate -- already 0/8 -- has no further room to improve and is
the headline confirmation that the burst hypothesis works.

Checkpoint used: `/scratch/gilbreth/wweligam/experiments/Motion/Wan22VJEPADynamics_stageA_burst3_2026_09_13_14_27_04/checkpoint-65000`.
Raw metrics JSON: kept in the session scratchpad (not yet copied to depot -- deferred until a more
complete/final checkpoint, per this project's "copy to depot on completion" convention; this is an
early interim checkpoint, not the finished Stage 1 run).

## checkpoint-70000: direction still 8/8 forward, quality flat/noisy vs. 65000 (2026-09-13)

```
Method                          PSNR    SSIM   LPIPS    EPE      FVD   n
Stage1 burst3 oracle ckpt65000  27.09  0.7424  0.0265  3.367   191.61  8
Stage1 burst3 oracle ckpt70000  26.95  0.7326  0.0260  3.736   208.53  8
```

Direction result is unchanged and remains the headline finding: **8/8 forward at both checkpoints**,
confirming this is not a one-off -- the burst input's non-time-symmetry reliably eliminates reversal
in oracle mode, well before quality has finished re-converging to the new (longer, higher-tubelet-
count) regime.

Quality metrics did not clearly improve from 65000->70000 (PSNR/SSIM/EPE/FVD all slightly worse,
LPIPS slightly better) -- likely within this eval's noise floor (n=8 scenes) rather than a genuine
regression; 5000 steps is a small fraction of the 20000-step budget. Continuing training toward
75000/80000 to see the fuller trend before drawing conclusions about final quality (per this
project's standing lesson not to over-read short-window trends either direction).

## First Stage-3 (deployable, chained) burst datapoint: still 0/8 reversal (2026-09-13)

Per the advisor's sequencing, chained the existing single-frame Stage B v2 predictor (checkpoint-50000,
never trained on burst data, never sees the real burst -- just the same single-image-to-tokens task it
was always trained on, applied independently to each of the 3 burst segments) with Stage 1's burst3
Wan checkpoint (checkpoint-70000). No new training was needed: `_dynamics_full_grid_burst`'s
`blur_jepa` branch already loops per segment through the unmodified predictor exactly like the
burst_size=1 case, so this ran on the first try (`Wan22VJEPADynamics_stageC_burst3_v2ckpt50000.yml`,
shape check passed immediately: seq_len=800 matched for dynamics_source=blur_jepa too).

```
Method                                              PSNR    SSIM   LPIPS    EPE      FVD   n
Stage1 burst3 oracle ckpt70000 (ceiling)            26.95  0.7326  0.0260  3.736   208.53  8
Stage3 burst3 chained (Stage1@70000 + v2ckpt50000)  25.74  0.6727  0.0343  4.102   222.81  8
```

**Direction: 8/8 forward again -- zero reversals, even with a fully deployable, no-oracle pipeline
using an off-the-shelf single-frame predictor never adapted for burst mode.** This is an important
result beyond just replicating the oracle finding: it shows the direction benefit comes from the
INPUT STRUCTURE (a temporally-ordered burst is not time-symmetric) rather than from oracle-quality
tokens -- even noisy/imperfect per-segment token predictions preserve enough relative temporal order
across segments to keep the whole generation pointed correctly forward. This is exactly the outcome
the burst redesign was meant to produce and the strongest evidence yet that it generalizes to the real
deployable setting, not just the oracle-validation setting.

Quality dropped somewhat from the oracle ceiling (PSNR -1.21, SSIM -0.0599, LPIPS +0.0083, EPE +0.366,
FVD +14.28) -- the same qualitative pattern as every single-frame Stage-A-oracle-vs-Stage-C-predictor
gap in this project's earlier work, and expected since this predictor (a) never sees the real burst
and (b) was never fine-tuned specifically for the burst regime. Two of the 8 scenes (IMG_0004a,
IMG_0183) show a visibly larger quality drop than the rest, suggesting per-segment prediction noise
compounds unevenly across scenes -- worth revisiting once/if a dedicated burst Stage 2 exists.

**Decision on dedicated Stage 2**: given (a) direction is already fully solved by the input structure
itself (0/8 reversal, oracle and chained alike) and (b) the off-the-shelf single-frame predictor
already produces a reasonable, coherent chained result with no new training, a dedicated burst-aware
Stage 2 finetune is not required to demonstrate the core hypothesis -- but remains a natural,
well-motivated next step to close the ~1-5% quality gap above (mirroring how the single-frame v2
predictor closed gaps to its own oracle over training). Deferred pending user direction on priority.

Checkpoints used: Stage 1 `.../Wan22VJEPADynamics_stageA_burst3_2026_09_13_14_27_04/checkpoint-70000`,
Stage B (predictor) `.../BlurJepaViT_stageB_vjepalora_v2_2026_09_10_14_08_22/checkpoint-50000/PeftModel_1`
(unchanged from every single-frame Stage C row above -- reused as-is).

## checkpoint-75000: quality recovers and improves, direction still 8/8 forward (2026-09-13)

```
Method                          PSNR    SSIM   LPIPS    EPE      FVD   n
Stage1 burst3 oracle ckpt65000  27.09  0.7424  0.0265  3.367   191.61  8
Stage1 burst3 oracle ckpt70000  26.95  0.7326  0.0260  3.736   208.53  8
Stage1 burst3 oracle ckpt75000  27.82  0.7583  0.0245  3.372   187.43  8
```

checkpoint-75000 beats both prior checkpoints on every metric except EPE (which is essentially tied
with 65000, and clearly better than 70000's dip) -- the 70000 dip looks like noise (n=8) rather than a
real regression, consistent with what this project already learned about not over-reading short
windows of a training-time signal. Direction remains 8/8 forward at all three checkpoints checked so
far (65000/70000/75000) -- zero reversals throughout, regardless of the quality wobble. One checkpoint
(80000, the end of the 20000-step budget) remains before Stage 1 finishes; expect this trend to
continue closing the gap to the single-frame oracle's own ceiling (PSNR=32.09/SSIM=0.8510/
LPIPS=0.0269/EPE=2.623/FVD=163.37 at 9 frames, i.e. before this run's harder 25-frame/burst target).

## Stage 1 completes its full 80000-step budget -- final oracle result, and a reversal-rate nuance in the chained pipeline (2026-09-14)

Stage 1 training finished its full 20000-step burst-adaptation budget (60000->80000) cleanly, no
crashes. Final oracle result:

```
Method                            PSNR    SSIM   LPIPS    EPE      FVD   n
Stage1 burst3 oracle ckpt65000    27.09  0.7424  0.0265  3.367   191.61  8
Stage1 burst3 oracle ckpt70000    26.95  0.7326  0.0260  3.736   208.53  8
Stage1 burst3 oracle ckpt75000    27.82  0.7583  0.0245  3.372   187.43  8
Stage1 burst3 oracle ckpt80000    27.78  0.7681  0.0242  2.647   199.66  8  <- final
Single-frame Stage A oracle (9f)  32.09  0.8510  0.0269  2.623   163.37  8  <- ceiling (easier task)
```

**Direction: 8/8 forward at every one of the 4 oracle checkpoints checked (65k/70k/75k/80k) --
zero reversals throughout the whole run.** EPE at the final checkpoint (2.647) is now essentially at
parity with the single-frame oracle's own ceiling (2.623, on the much easier 9-frame/5-tubelet task),
and LPIPS (0.0242) actually beats it (0.0269) -- remarkable given this run reconstructs 3x more real
motion content (25 frames from 15 tubelets vs. 9 frames from 5). PSNR/SSIM/FVD remain behind the
single-frame ceiling, consistent with the harder task and the same total (20000-step) adaptation
budget.

**However, the chained (deployable, Stage 3) pipeline shows a nuance the oracle numbers alone hide.**
Re-running the Stage 3 chained eval (existing single-frame v2 checkpoint-50000 predictor, never
retrained for burst) against this FINAL Stage 1 checkpoint:

```
Method                                                PSNR    SSIM   LPIPS    EPE      FVD   n  reversed
Stage3 burst3 chained @ Stage1 ckpt70000              25.74  0.6727  0.0343  4.102   222.81  8  0/8
Stage3 burst3 chained @ Stage1 ckpt80000 (final)       25.92  0.6807  0.0346  4.152   277.19  8  2/8
```

At the final checkpoint, 2 of 8 scenes (GOPR9637a, IMG_0023) score marginally better reversed instead
of forward -- both are borderline cases where the forward/reversed PSNR gap is small either way
(GOPR9637a: 28.33 reversed vs. essentially the same magnitude forward at ckpt70000 it was 28.46
forward; IMG_0023 similarly 25.21 vs. 25.56 previously), so this reads more like a couple of
close-call scenes flipping classification than a systematic new failure mode -- overall quality
(PSNR/SSIM/LPIPS/EPE) is not meaningfully different between the two chained checkpoints, only FVD is
clearly worse (277.19 vs. 222.81).

**Working hypothesis, not yet confirmed**: as Stage 1 continues training purely on real oracle V-JEPA
tokens, it may be specializing somewhat to that exact (in-distribution) token statistic, making it
marginally LESS robust to the different noise characteristics of the single-frame predictor's
out-of-distribution tokens -- i.e. the oracle pipeline's continued improvement (which is real and
monotonic on direction) does not necessarily transfer 1:1 to the chained pipeline's robustness once the
predictor is swapped in. This is exactly the kind of gap a dedicated burst-aware Stage 2 fine-tune
(training Stage 1 and the predictor jointly, or at least on matched token statistics) would be
expected to close -- strengthening the earlier "optional but well-motivated" framing of that next
step into "worth prioritizing if quality is now the ask, though direction itself is still solved
overwhelmingly (6/8 to 8/8 forward at every chained checkpoint checked, vs. the single-frame line's
2/8-5/8 ceiling)."

**Bottom line, unchanged from earlier and still the headline result**: burst conditioning has
converted a coin-flip reversal problem (2/8-5/8 forward, no fix found across an entire prior line of
work) into an overwhelmingly-forward one (6/8-8/8 forward across every burst checkpoint checked, oracle
and chained alike) purely from input structure, not from token quality. The one-checkpoint dip to 6/8
at the very end of training is a real observation worth tracking, not a reason to doubt the core
finding.

## Stage 2 (burst-aware V-JEPA predictor finetune) launched -- caught and fixed a real OOM bug (2026-09-14)

Extended `BlurJepaDynamics_model.py` with the same additive `burst_size` pattern as the Wan model:
`_optimize_parameters_burst`/`_validate_step_burst` loop the existing per-image predict/target/loss
logic once per burst segment (no new architecture or loss -- see the class docstring), warm-started
from the single-frame Stage B v2 checkpoint-50000 into a new experiment folder
(`BlurJepaViT_stageB_burst3_v2ckpt50000.yml`), extending to a 70000-step budget (50000+20000, matching
this project's established pattern). Smoke-tested cleanly at batch_size=2 first.

**First real-training attempt at batch_size=8 (matching the single-frame recipe) hit a genuine CUDA
OOM within the first step**, despite this predictor being far smaller than Wan-5B. Root cause: the
first implementation collected all `burst_size` segments' per-sample losses into a list and called
`accelerator.backward()` ONCE on their combined mean -- this holds all `burst_size` segments' full
forward computation graphs in memory SIMULTANEOUSLY until that single backward call, roughly
multiplying peak activation memory by `burst_size` versus the single-frame case, rather than the naive
"3x more images per step" estimate one might expect. Fixed by backing up per-segment instead
(gradient-accumulation style): call `accelerator.backward(seg_loss.mean() / burst_size)` immediately
after each segment's loss is computed, accumulating gradients across the `burst_size` backward calls
before one `optimizer.step()` -- mathematically identical gradient (linearity of differentiation) to
a single combined backward, but autograd frees each segment's graph immediately after its own backward
call, so peak memory returns to the same footprint as burst_size=1 regardless of burst_size.

**General lesson for this codebase**: whenever training code loops over multiple independent
"sub-samples" per training example (segments, views, augmentations, etc.) and combines their losses,
backward-ing per-sub-sample (scaled by 1/N, accumulated) is the memory-safe pattern -- collecting N
loss terms before one combined backward is a real, easy-to-miss OOM risk that scales with N, distinct
from (and easy to conflate with) the ordinary "N times the data per step" compute/memory scaling one
naturally expects.

Relaunched successfully after the fix -- resumed cleanly from checkpoint-50000, stable memory (~28.6GB
for this job on top of an unrelated ~32GB co-located job on the same shared node, well within the
82GB budget), though throughput is markedly slower than the single-frame recipe (~15s/step vs. ~2s/step)
due to the sequential per-segment backward plus heavy contention from that co-located job at 100% GPU
utilization. Training in progress toward the 70000-step budget.

## First interim Stage 3 result with the burst-finetuned predictor (2026-09-15)

Ran the first successful interim Stage 3 (chained, deployable) eval using Stage 2's burst-aware
predictor finetune (checkpoint-58000, ~8000 steps into its 20000-step burst-specific budget) instead
of the original off-the-shelf single-frame v2 checkpoint-50000, paired with the same Stage 1
checkpoint-80000 as before -- an apples-to-apples comparison of predictor-only change.

Two earlier interim attempts (checkpoints 53000, 56500) were skipped/failed due to insufficient GPU
headroom on the shared node (an unrelated co-located job, `VAEBlurAlignFluxLoRA_DP_GoPro_arch1024_longrun`,
was consuming ~33GB); that job was explicitly stopped by the user, freeing ~53GB and also nearly
tripling Stage 2's own training throughput (4.3s/it vs. 10-15s/it under contention).

```
Method                                                      PSNR    SSIM   LPIPS    EPE      FVD   n  reversed
Chained @ Stage1 ckpt70000 + v2ckpt50000 (unfinetuned)      25.74  0.6727  0.0343  4.102   222.81  8  0/8
Chained @ Stage1 ckpt80000 + v2ckpt50000 (unfinetuned)      25.92  0.6807  0.0346  4.152   277.19  8  2/8
Chained @ Stage1 ckpt80000 + Stage2-burst ckpt58000 (NEW)   25.95  0.6805  0.0346  4.469   253.89  8  1/8
```

Comparing the two rows that hold Stage 1 fixed at checkpoint-80000 (isolating the predictor swap):
PSNR/SSIM/LPIPS are essentially unchanged (within noise), EPE is slightly worse (+0.317), FVD improved
(-23.30), and the reversal count improved from 2/8 to 1/8 (only IMG_0023 reversed, the same scene that
was one of the two reversed cases at ckpt80000 before). This is a modest, mixed result rather than a
dramatic win -- reasonable given Stage 2's burst-specific finetune is only ~40% through its own
20000-step budget (checkpoint-58000 of a 70000 target) at this checkpoint. Will re-check as Stage 2
approaches completion; the working hypothesis from the prior entry (co-adapting predictor and Wan-side
training should close the chained-vs-oracle gap and stabilize the one observed reversal case) remains
to be confirmed or refuted by the fuller trend.

## checkpoint-59000: within-noise fluctuation over the last 1000 steps (2026-09-15)

```
Method                                                      PSNR    SSIM   LPIPS    EPE      FVD   n  reversed
Chained @ Stage1 ckpt80000 + Stage2-burst ckpt58000         25.95  0.6805  0.0346  4.469   253.89  8  1/8
Chained @ Stage1 ckpt80000 + Stage2-burst ckpt59000         25.96  0.6808  0.0345  4.667   239.61  8  2/8
```

Only 1000 Stage 2 steps apart (this eval ran shortly after the 58000 check, per the scheduled 6-hour
loop). PSNR/SSIM/LPIPS essentially unchanged, EPE slightly worse, FVD improved, reversal count went
from 1/8 to 2/8 (GOPR9637a newly reversed alongside IMG_0023, both borderline cases in the same
family as every prior chained-checkpoint reversal seen in this line of work). Reads as ordinary
n=8 eval noise rather than a real regression -- not enough new training to draw a trend conclusion
yet. Stage 2 training itself is now running ~3x faster since the co-located contending job was
stopped (~4.3s/it), so the next scheduled check should cover substantially more steps.

## Stage 2 checkpoint-64000 interim result, and the full trend so far (2026-09-15)

```
Stage2 ckpt   PSNR    SSIM   LPIPS    EPE      FVD   n  reversed
58000        25.95  0.6805  0.0346  4.469   253.89  8  1/8
59000        25.96  0.6808  0.0345  4.667   239.61  8  2/8
64000        25.97  0.6830  0.0341  4.206   251.29  8  1/8 (IMG_0023 only)
```

All against the same fixed Stage 1 checkpoint (80000). PSNR/SSIM/LPIPS show a slow, gentle, roughly
monotonic improvement across these three checkpoints (small enough to be partly noise, but
consistently in the same direction each time); EPE/FVD/reversal fluctuate within a narrow band with
no clear trend either way. IMG_0023 has now reversed at every chained checkpoint checked in this
whole burst-predictor-finetune line -- the one persistently borderline scene.

**Decision: resume with LR decay once this run completes.** The training-time loss/flip_rate has
looked essentially flat for most of this run (same pattern the single-frame v2->100k extension showed
before its own real, if delayed, quality gains -- see the 2026-09-10/12 entries above), and the
interim Stage 3 numbers above show only slow, modest movement -- consistent with a shallow
optimization plateau that a decaying LR could help escape. `BlurJepaViT_stageB_burst3_v2ckpt50000_lrdecay.yml`
(cosine decay 1e-4->0 over steps 70000->90000, resuming into a new experiment folder) is prepared and
verified -- see that file's own header comment for the empirical repro proving the schedule-type swap
takes effect correctly across accelerate's checkpoint resume (a distinct, verified-safe case from the
documented `learning_rate`-scalar resume no-op). Will launch automatically once Stage 2's current
constant-LR run reaches checkpoint-70000.

## Full hires (1280x736) comparison: burst vs. every established baseline and the single-frame line (2026-09-15)

Per explicit user instruction: "check the results" now always means the hires (1280x736) check, and
this comparison must include every established baseline plus the single-frame line, not just prior
burst checkpoints.

```
Method                                          PSNR    SSIM   LPIPS     EPE      FVD   reversed
Tedla et al. 2025 (external baseline)           28.40  0.7961  0.0234   5.034  1357.17   --
Content-blind ablation                          30.59  0.8066  0.0316   5.004   262.17   --
Stage C, from-scratch predictor (single-image)  30.71  0.8101  0.0280   5.916   293.44   --
Stage C, vjepa_lora v1 (single-image, 20k)      31.62  0.8374  0.0256   5.061   216.00   6/8
Stage C, vjepa_lora v2 (single-image, 20k)      31.79  0.8415  0.0257   4.593   195.43   4/8
Stage C, vjepa_lora v2 (single-image, 50k)      31.98  0.8460  0.0250   3.310   168.80   5/8
Stage A oracle (single-image, ceiling)          32.09  0.8510  0.0269   2.623   163.37   --
Stage 1 burst3 oracle (ckpt80000, FINAL)        29.37  0.7613  0.0363   8.559   304.23   0/8
Stage 3 burst3 chained (Stage1@80000+ckpt64500) 28.08  0.7225  0.0405  10.253   347.55   1/8
```

**Direction: the burst approach's central claim holds decisively at hires too.** Every single-image
checkpoint across the whole project's history sat at 4/8-6/8 reversed (i.e. never better than a
coin flip) no matter how much training or loss-shaping was applied -- the burst oracle gets 0/8
reversed (8/8 forward) and the fully deployable chained pipeline gets 1/8 reversed (7/8 forward,
only IMG_0179 -- notably a DIFFERENT scene than the persistently-borderline IMG_0023 seen at native
resolution, suggesting these are close-call scenes rather than a systematic failure mode).

**Quality: the burst approach currently trails the single-image line on raw fidelity metrics at
hires**, honestly reported: PSNR (28.08 vs. 31.98), EPE (10.253 vs. 3.310), and FVD (347.55 vs.
168.80) are all meaningfully worse than the single-image v2(50k) checkpoint. This is not a
surprising or alarming result -- the burst pipeline generates a 25-frame video (vs. 9 frames for the
single-image line), reconstructing roughly 3x more real motion content, and Stage 2 (the burst
predictor) is still mid-training (65k of a 70k-step budget, with the further LR-decay refinement not
yet applied) at the time of this check. The honest comparison is: burst trades some current fidelity
for a structural fix to the reversal problem that no amount of single-image-line training or
loss-shaping ever solved -- whether the fidelity gap closes further is what the remaining Stage 2
budget + LR-decay follow-up will show.

Depot copy deferred until Stage 2 finishes (checkpoint-70000) and, if it helps, the LR-decay
follow-up -- per this project's "copy to depot on completion" convention, this is still an interim
checkpoint, not the finished Stage 2 run.

## checkpoint-68500: a real, decisive improvement (not noise) as the constant-LR budget nears its end (2026-09-16)

```
Stage2 ckpt   PSNR    SSIM   LPIPS    EPE      FVD   n  reversed
58000        25.95  0.6805  0.0346  4.469   253.89  8  1/8
59000        25.96  0.6808  0.0345  4.667   239.61  8  2/8
64000        25.97  0.6830  0.0341  4.206   251.29  8  1/8
68500        26.39  0.7096  0.0316  3.958   237.48  8  1/8 (IMG_0023 only)
```

Unlike the 58000->59000 fluctuation (a mixed bag, read as noise), every metric moved together in the
improving direction from 64000->68500 (PSNR +0.42, SSIM +0.027 -- the largest single jump in this
whole interim series, LPIPS/EPE/FVD all better). Reversal holds steady at 1/8 (IMG_0023 only).
Training is at step 68601/70000 (98%) at time of writing, ETA ~2 hours to complete its 70000-step
constant-LR budget; the LR-decay follow-up (`BlurJepaViT_stageB_burst3_v2ckpt50000_lrdecay.yml`) will
launch automatically once it does.

## Stage 2 constant-LR run completed; LR-decay follow-up launched successfully (2026-09-16)

Constant-LR run (`BlurJepaViT_stageB_burst3_v2ckpt50000.yml`) reached its full 70000-step budget
cleanly (process exited, no crash). Immediately launched the LR-decay follow-up
(`BlurJepaViT_stageB_burst3_v2ckpt50000_lrdecay.yml`) into a new experiment folder:
`/scratch/gilbreth/wweligam/experiments/Motion/BlurJepaViT_stageB_burst3_v2ckpt50000_lrdecay_2026_09_16_02_27_24/`.

Verified the resume worked exactly as the empirical repro predicted (see this run's config header):
resumed cleanly from checkpoint-70000, `lr_0` reads 0.0001 (not an erroneous near-zero jump) at every
step checked so far (through step 70186, ~1% into the 20000-step cosine decay window -- correctly
still near-flat this early, matching cosine decay's shape). Training running at ~4.28s/it, no GPU
contention. Budget: 70000 -> 90000 steps.

## First LR-decay interim check (hires, ckpt73000): a real regression across every metric (2026-09-16)

```
Method                                                      PSNR    SSIM   LPIPS     EPE      FVD  reversed
Chained hires @ Stage2 ckpt64500 (pre-decay)                28.08  0.7225  0.0405  10.253   347.55  1/8
Chained hires @ Stage2-lrdecay ckpt73000 (2500 into decay)  27.41  0.6975  0.0430  11.336   365.00  2/8
```

Every single metric moved in the WORSE direction (PSNR -0.67, SSIM -0.025, LPIPS +0.0025, EPE +1.08,
FVD +17.45), and reversal got worse too (1/8 -> 2/8, GOPR9637a newly reversed alongside IMG_0179).
This is honestly reported as a real regression, not dismissed as noise -- every metric moving the
same direction together is a different signature than the earlier 58000->59000 fluctuation (a mixed
bag, correctly read as noise at the time).

**Caveat that matters here**: this comparison's "before" point is checkpoint-64500 (the last hires
check taken during the constant-LR phase), not checkpoint-70000 itself (the actual point the decay
started from) -- no hires check was taken exactly at 70000. The native trend (58000->68500) showed
continued, sometimes substantial, improvement right up to the end of the constant-LR budget (see the
2026-09-16 checkpoint-68500 entry above), so the TRUE starting point for the decay was very likely
better than what checkpoint-64500 represents -- meaning this regression, measured against the actual
70000 starting point, could be even larger than the numbers above suggest.

**Working hypothesis, not yet confirmed**: an early, temporary dip right after switching from constant
to decaying LR is a known pattern in some optimization landscapes (the model "unsettles" from the
constant-phase optimum before the decaying schedule guides it to a better one) -- this would predict
recovery and eventual improvement as the decay progresses further. The alternative, more concerning
hypothesis is that the decay schedule itself is not helping this recipe. Continuing to track via the
6-hour interim checks; will not conclude either way from this single data point.

## LR-decay checkpoint-77500: the regression is continuing, not a temporary dip (2026-09-16)

```
Method                                                      PSNR    SSIM   LPIPS     EPE      FVD  reversed
Chained hires @ Stage2 ckpt64500 (pre-decay)                28.08  0.7225  0.0405  10.253   347.55  1/8
Chained hires @ Stage2-lrdecay ckpt73000 (2500 into decay)  27.41  0.6975  0.0430  11.336   365.00  2/8
Chained hires @ Stage2-lrdecay ckpt77500 (7500 into decay)  26.90  0.6802  0.0478  10.340   348.36  2/8
```

Three consecutive checks now (5000 steps apart) show PSNR/SSIM/LPIPS declining MONOTONICALLY since the
decay began (PSNR: 28.08->27.41->26.90; SSIM: 0.7225->0.6975->0.6802; LPIPS: 0.0405->0.0430->0.0478).
EPE/FVD are roughly flat/noisy rather than trending. Reversal held at 2/8 (same two scenes, GOPR9637a
and IMG_0179, both times).

**This rules out the "temporary dip before recovery" hypothesis from the checkpoint-73000 entry --
three points on a smooth monotonic decline is a real trend, not noise settling out.** The alternative
hypothesis (the decay schedule itself is actively hurting this recipe, not just failing to help) is
now the better-supported reading of the data. Flagged clearly to the user rather than waiting for
further scheduled checks to keep confirming the same declining trend.

## Stage 2 (burst predictor) work paused; root cause redirected to Stage 1 (2026-09-16)

Per user direction, stopped the Stage 2 LR-decay run (checkpoint-77500, PID confirmed gone, GPU freed)
-- the constant-LR run's checkpoint-70000/checkpoint-68500 is considered converged/good-enough for now.
User diagnosis, confirmed against the actual configs: Stage 1 burst's warm-start checkpoint
(`Wan22VJEPADynamics_stageA_lora_deblur_ar.yml`, checkpoint-60000) was trained WITH
`aspect_ratio_range: [0.5, 2.0]` specifically for resolution/aspect-ratio generalization, but burst
mode's entire 20000-step adaptation phase (`Wan22VJEPADynamics_stageA_burst3.yml`) trained on ONLY
fixed 320x320 square crops (aspect_ratio_range was explicitly unsupported for burst_size>1) --
silently eroding that generalization while also learning a 3x-harder task in the same budget. This
plausibly explains both the observed hires motion inconsistency and the large native->hires EPE gap
(2.647->8.559, ~3.2x) seen at Stage 1's own final checkpoint.

**Plan (in progress)**:
1. Implement `aspect_ratio_range` support for burst_size>1 (dataset-level, extending the existing
   deferred-crop mechanism to burst-shaped batches).
2. Fix `vjepa_from_unit_tensor` (`models/LucidFlux/jepa_helpers.py`) to letterbox (aspect-preserving
   resize + zero-pad) non-square input instead of stretching it to a square `vjepa_image_size` --
   applies uniformly to every V-JEPA caller in this project (oracle and predictor, single-frame and
   burst); square input (the historical crop_size=320 default) is unaffected, byte-identical to
   before.
3. Retrain Stage 1 burst with aspect_ratio_range restored, using a cosine LR decay with a FLOOR of
   2e-5 (not decaying to 0), continuing until forward-direction PSNR reaches 31dB+.

## Aspect-ratio fix implemented, verified, and launched (2026-09-16)

Implemented and smoke-tested all three pieces of the fix:
1. `dataset/adobe240_dataset.py`: extended the existing deferred-crop/collate_fn mechanism to
   burst-shaped batches (`_native_burst_blur01`/`_native_burst_full_frames01`), removing the prior
   NotImplementedError -- burst_size>1 now supports aspect_ratio_range exactly like the single-frame
   path (one shared non-square crop per training batch, always square for validation).
2. `models/LucidFlux/jepa_helpers.py::vjepa_from_unit_tensor`: now letterboxes (aspect-preserving
   resize + exact-zero padding, verified via a direct unit test) non-square input instead of
   stretching it to a square vjepa_image_size -- applies uniformly to every V-JEPA caller (oracle and
   predictor, single-frame and burst). Square input (crop_size=320, the historical default) is a
   byte-identical no-op.
3. `models/base_model.py`: new `cosine_floor` scheduler type (LambdaLR built directly, since
   diffusers.get_scheduler has no floor parameter) -- decays from `learning_rate` to an absolute
   `lr_floor` and then holds there, rather than continuing to exactly 0. Verified with a standalone
   repro (warmup->decay->exact floor->holds past the schedule's end) before use.

Combined smoke test (`Wan22VJEPADynamics_stageA_burst3_ar_smoke.yml`) confirmed all three work
together: `lr_0` sequence 1e-4->8e-5->4e-5->2e-5 over 4 steps landed exactly on the configured floor,
no shape errors from the new non-square burst+aspect-ratio dataloader path.

Launched the real run (`Wan22VJEPADynamics_stageA_burst3_ar.yml`): resumed cleanly from the completed
burst3 run's checkpoint-80000 into a new experiment folder
(`Wan22VJEPADynamics_stageA_burst3_ar_2026_09_16_15_03_14`), `lr_0` correctly reads 0.0001 at the
resume point (not an erroneous jump -- confirms `lr_warmup_steps=80000` being set to the absolute
resume step, not 0, worked correctly here too). Budget: 80000->140000 steps, but per explicit
instruction this is an open-ended run to be stopped once hires forward-direction PSNR reaches 31dB+,
checked periodically via the usual out-of-band bidirectional-metrics script (not the native-only
in-loop validation).

## First interim check on Stage A aspect-ratio-fix run (checkpoint-87000, hires) (2026-09-16)

7000 steps into the aspect-ratio-fix run (80000->87000, budget continues to 140000, cosine_floor decay
active, lr_0=9.73e-5 at time of this checkpoint).

```
Method                                      PSNR    SSIM   LPIPS     EPE      FVD   reversed
Stage1 burst3 oracle hires ckpt80000 (pre-fix)  29.37  0.7613  0.0363   8.559   304.23   0/8
Stage1 burst3-ar oracle hires ckpt87000         29.35  0.7633  0.0355   7.760   272.35   1/8
```

Mean forward-direction-only PSNR (excluding the one reversed scene, IMG_0015): **29.13 dB** -- still
well short of the 31dB+ target. Aggregate PSNR/SSIM/LPIPS/FVD are all roughly flat-to-slightly-better
than the pre-fix checkpoint-80000, and EPE improved (8.559->7.760). Direction ticked from 0/8 to 1/8
reversed (IMG_0015) -- a small step backward on direction, but not concerning at this stage: only 7000
steps in, the model is still absorbing the newly-reintroduced aspect_ratio_range training signal (a
genuinely new augmentation it saw zero of during its entire prior 20000-step burst-adaptation phase),
and one flipped scene out of 8 is well within normal run-to-run variation seen throughout this project.

No red flags: nothing regressing sharply, no errors, LR decaying on schedule. Continuing to track
via the 6-hour recurring check; 31dB is still a distant target from the current ~29.1-29.4dB range,
so expect several more check-ins before (if) it's reached.

## Added V-JEPA dynamics-token PCA visualization to the validation grid (2026-09-16)

Per user request: `Wan22VJEPADynamics_model.py`'s `validate_step` now logs a 2x2 diagnostic grid
(under the same `wan_pred_vs_gt_{stem}` video name as before) instead of the old 2-row pred/gt stack:
- top-left: predicted video (unchanged from before)
- top-right: **new** -- V-JEPA dynamics-token PCA visualization (`_pca_rgb`/`_vjepa_pca_video_thwc`):
  fits PCA (via SVD) on every spatial position of every tubelet in the sample's full dynamics grid
  jointly (so the RGB color mapping is consistent across time, not refit per frame), maps the top-3
  principal components to RGB, then nearest-upsamples in time (vjepa_num_tubelets -> n_frames) and
  space (vjepa_grid_h x vjepa_grid_w -> the video's H,W) to align with the other panels. Works for
  both single-frame and burst mode (burst's concatenated tubelet-axis grid shows all segments back to
  back in the correct order, exactly matching `_dynamics_full_grid_burst`'s construction).
- bottom-left: ground-truth video (unchanged from before)
- bottom-right: **new** -- GT optical-flow visualization (reuses the already-computed `gt_flow_thwc`,
  previously only logged standalone -- now also reused here rather than recomputed)

Verified: a standalone unit test of the PCA/upsampling math (correct shapes, [0,1] range, correct
temporal nearest-mapping for both single-frame 5-tubelet and burst 15-tubelet cases), then a full
smoke test through the real trainer (`Wan22VJEPADynamics_stageA_burst3_ar_smoke.yml`) confirmed no
shape errors and the same video/image upload counts as before the change. Visually inspected the
actual output frame: all four quadrants align correctly with no distortion, and the PCA panel shows
genuine spatial structure (not a uniform blank), confirming the PCA computation is operating on real,
differentiated token content.

## checkpoint-99500 (hires): direction back to perfect, quality improving toward the 31dB target (2026-09-17)

19000 steps into the aspect-ratio-fix run (80000->99500 of the 140000 budget), lr_0=8.15e-5 (per
cosine_floor schedule).

```
Method                                       PSNR    SSIM   LPIPS     EPE      FVD   reversed
Stage1 burst3 oracle hires ckpt80000 (pre-fix)   29.37  0.7613  0.0363   8.559   304.23   0/8
Stage1 burst3-ar oracle hires ckpt87000          29.35  0.7633  0.0355   7.760   272.35   1/8
Stage1 burst3-ar oracle hires ckpt99500          29.47  0.7644  0.0352   7.768   270.42   0/8
```

**Mean forward-direction-only PSNR: 29.47 dB** (all 8 scenes forward, so this equals the aggregate) --
up from 29.13dB at checkpoint-87000. Direction returned to a clean 0/8 reversed (the single flipped
scene at checkpoint-87000, IMG_0015, is back to forward). PSNR/SSIM/LPIPS all improved modestly;
EPE/FVD roughly flat. Still short of the 31dB+ target (gap: 1.53dB), but moving in the right direction
on every axis with no red flags. Continuing to track via the 6-hour recurring check.

## checkpoint-111500 (hires): steady progress toward the 31dB target (2026-09-17)

31500 steps into the aspect-ratio-fix run (80000->111500 of the 140000 budget), lr_0 was 5.71e-5 at
launch of this check (training has since progressed further, currently at step 119438, lr_0=4.1e-5).

```
Stage2 ckpt (Stage 1, hires)                  PSNR    SSIM   LPIPS     EPE      FVD   reversed
ckpt87000                                     29.35  0.7633  0.0355   7.760   272.35   1/8
ckpt99500                                     29.47  0.7644  0.0352   7.768   270.42   0/8
ckpt111500                                    29.81  0.7732  0.0344   8.515   268.99   0/8
```

**Mean forward-direction PSNR: 29.81 dB** (all 8 scenes forward again) -- up from 29.47dB, continuing
the improving trend. Gap to the 31dB+ target has narrowed to 1.19dB. SSIM/LPIPS/FVD also improved;
EPE ticked up slightly (within the noise band already seen at n=8). Direction has now held at a clean
0/8 reversed for two consecutive checks. No red flags -- steady, real progress toward the target.

## checkpoint-124000 (hires): small dip, not a clean continuation of the trend (2026-09-17)

12500 steps further (111500->124000 of the 140000 budget). lr_0 has decayed further under the
`cosine_floor` schedule (approaching, but not yet at, the 2e-5 floor).

```
Stage2 ckpt (Stage 1, hires)                  PSNR    SSIM   LPIPS     EPE      FVD   reversed
ckpt87000                                     29.35  0.7633  0.0355   7.760   272.35   1/8
ckpt99500                                     29.47  0.7644  0.0352   7.768   270.42   0/8
ckpt111500                                    29.81  0.7732  0.0344   8.515   268.99   0/8
ckpt124000                                    29.64  0.7707  0.0347   8.136   296.01   0/8
```

**Mean forward-direction PSNR: 29.64 dB** (all 8 scenes still forward, 0/8 reversed -- direction quality
is holding fine). This is a **0.17dB dip from ckpt111500's 29.81dB**, not an improvement -- flagging this
honestly rather than reporting only the favorable trend. SSIM/LPIPS are both essentially flat (0.7707 vs
0.7732, 0.0347 vs 0.0344) and FVD moved up somewhat (296.01 vs 268.99), while EPE improved slightly
(8.136 vs 8.515). With n=8 scenes and per-scene PSNR spanning 25.53-33.91dB, a 0.17dB swing in the mean
is well within the sampling noise already observed between the two prior "improving" steps (99500->111500
was itself a 0.34dB move) -- this reads as a plateau around ~29.5-29.8dB with noise, not a reversal of the
underlying trend, but it is also not further progress. Gap to the 31dB+ target is still ~1.4dB and has not
meaningfully closed since ckpt111500. Continuing to monitor; will flag more decisively if the next 1-2
checks also fail to exceed 29.81dB.

## checkpoint-136000 (hires): back to the ckpt111500 level; budget nearly exhausted (2026-09-17)

11500 steps further (124000->136000 of the 140000 budget). lr_0=2.05e-5 at time of this check --
essentially at the `cosine_floor` floor of 2e-5 already.

```
Stage2 ckpt (Stage 1, hires)                  PSNR    SSIM   LPIPS     EPE      FVD   reversed
ckpt87000                                     29.35  0.7633  0.0355   7.760   272.35   1/8
ckpt99500                                     29.47  0.7644  0.0352   7.768   270.42   0/8
ckpt111500                                    29.81  0.7732  0.0344   8.515   268.99   0/8
ckpt124000                                    29.64  0.7707  0.0347   8.136   296.01   0/8
ckpt136000                                    29.80  0.7737  0.0340   8.221   245.72   0/8
```

**Mean forward-direction PSNR: 29.80 dB** (all 8 scenes forward, 0/8 reversed -- direction quality has
now held clean for four consecutive checks). This essentially recovers the ckpt124000 dip and matches
ckpt111500's 29.81dB -- so the previous entry's "plateau with noise" read looks correct in hindsight,
not an early regression. SSIM/LPIPS/FVD are all the best-yet in this series. Gap to the 31dB+ target is
still ~1.2dB and has **not meaningfully closed since ckpt111500** (three checks spanning 12500 training
steps have oscillated in a ~29.6-29.8dB band without net progress) -- this now looks more like a genuine
plateau near the LR floor than continuing improvement, worth flagging plainly rather than only citing the
best number in the series.

**Budget alert:** training is at step 137000/140000 (lr_0=2.05e-5, i.e. already at the floor) -- only
~3000 steps (roughly 1.5-2h) remain before this run hits `max_train_steps` and stops on its own. The
31dB+ target has not been reached, and the last three checks suggest the run may already be near its
achievable ceiling under the current schedule/floor rather than still climbing. Flagging this now so the
budget can be extended (or the plateau accepted) before the run auto-stops, rather than discovering it
after the fact.

## checkpoint-140000 (hires): run completed its full budget; 31dB target NOT reached (2026-09-18)

Training reached its full `max_train_steps=140000` budget and exited cleanly on its own (process gone,
GPU fully freed, lr_0 landed exactly at the 2e-5 floor as designed, no crash/OOM in the log). **This run
is no longer active** -- there is nothing left to "continue" under this config until a new budget is
launched.

```
Stage2 ckpt (Stage 1, hires)                  PSNR    SSIM   LPIPS     EPE      FVD   reversed
ckpt87000                                     29.35  0.7633  0.0355   7.760   272.35   1/8
ckpt99500                                     29.47  0.7644  0.0352   7.768   270.42   0/8
ckpt111500                                    29.81  0.7732  0.0344   8.515   268.99   0/8
ckpt124000                                    29.64  0.7707  0.0347   8.136   296.01   0/8
ckpt136000                                    29.80  0.7737  0.0340   8.221   245.72   0/8
ckpt140000 (FINAL, budget exhausted)          29.94  0.7763  0.0338   7.456   273.96   0/8
```

**Mean forward-direction PSNR: 29.94 dB** (all 8 scenes forward, 0/8 reversed -- clean direction across
all five checks since the aspect-ratio fix landed). This is the best PSNR/SSIM/LPIPS/EPE in the whole
`_ar` series (EPE's best-yet at 7.456, a genuine improvement over the 8.1-8.5 band the last three checks
sat in), though FVD is mid-pack rather than the best. **The 31dB+ target was NOT reached** -- final gap
is 1.06dB. Reading the full trajectory (87000->140000: 29.35->29.47->29.81->29.64->29.80->29.94), the
`_ar` fix produced a real, if noisy, net improvement of +0.59dB over 53000 steps and a full elimination
of the reversal problem (1/8->0/8, held for the last four checks) -- but growth has clearly slowed as lr
approached the floor, and nothing here suggests the model would have blown past 31dB with a bit more of
the *same* schedule. Whether to extend with a fresh budget (same or extended cosine_floor decay, or a
different LR) is a call for the user, not something to decide unilaterally.

## Status check + depot backup + known-motion dataset survey for oracle upper-bound testing (2026-09-21)

**Training status:** nothing is currently running. Both allocated A100s (gilbreth-k024, gilbreth-k026)
are idle interactive shells with zero GPU compute processes -- no python/accelerate process on either.
Stage A (oracle, `Wan22VJEPADynamics_stageA_burst3_ar`) is **done**: completed its full 140000-step
budget on 2026-09-17/18 at 29.94dB hires forward-PSNR (see entry directly above), no further steps will
happen under that config. Stage B (`BlurJepaViT_stageB_burst3_v2ckpt50000_lrdecay`) is **paused**, last
checkpoint dated 2026-09-16 14:21 -- per the "Stage 2 work paused; root cause redirected to Stage 1"
entry, this was intentionally stopped mid-regression-investigation and never resumed after the Stage 1
aspect-ratio fix landed; it is not "in progress," it is idle/abandoned pending a decision to restart it
against the new Stage 1 checkpoint.

**Depot backup:** copied `Wan22VJEPADynamics_stageA_burst3_ar_2026_09_16_15_03_14/checkpoint-140000`
(124M) plus its launch config to
`/depot/chan129/users/harshana/backups/Wan22VJEPADynamics_stageA_burst3_ar_2026_09_16_15_03_14/` --
this is the actual "Stage A oracle" upper-bound checkpoint the new experiments below will use. Also
audited and backed up **Stage B's constant-LR endpoint**,
`BlurJepaViT_stageB_burst3_v2ckpt50000_2026_09_14_11_18_08/checkpoint-70000` (75M + config), to
`/depot/chan129/users/harshana/backups/BlurJepaViT_stageB_burst3_v2ckpt50000_2026_09_14_11_18_08/` --
this is a completed run (per the earlier "Stage 2 v2 extended to 100k steps, then stopped at
checkpoint-50000" / "constant-LR run completed" entries) whose checkpoint-70000 produced this project's
best Stage 3 chained numbers, so it's the real non-oracle half of the deployable pipeline and was worth
freezing per [[scratch_45day_rolling_purge]] even though this session's focus is the oracle test. A
find-based weight-file audit confirmed both this checkpoint and the paused `lrdecay` run's
checkpoint-79500 still have their weights intact (not yet hit by the scratch purge) -- did NOT back up
the `lrdecay` checkpoints since that run regressed every metric relative to checkpoint-70000 and was
paused rather than completed; nothing there is worth freezing over the constant-LR endpoint. Also did
not back up the pre-fix `stageA_burst3` (ckpt80000) run since it's strictly superseded by the `_ar`
run's ckpt87000+ series. Depot (`/depot/chan129`) is at 96% capacity (547G free) -- keep backups to
final/selected checkpoints, not whole experiment directories.

**Important framing correction:** 29.94dB is the in-distribution Adobe240-val number, not an "upper
bound" in the sense the new goal below is asking about -- the open question is how the oracle model
generalizes to recognizable real-world motions outside that training distribution, which the existing
table doesn't speak to at all.

**New session goal:** find the oracle Stage A model's upper bound by feeding it real, recognizable
physical motions (ball bouncing, object falling, etc.) instead of the training-distribution burst
crops, and check qualitatively whether the generated video looks physically correct. This needs (a) a
source of real video with clean, well-known, simple physical motions and (b) ideally high native fps so
a realistic burst-of-blur-crops can be synthesized the same way training data was (temporal averaging of
consecutive high-fps frames, mirroring the Adobe240fps pipeline in `dataset/adobe240_dataset.py`).

**The oracle constraint rules two candidates out.** Stage A oracle computes V-JEPA dynamics tokens from
the *ground-truth sharp future frames themselves* (that's what "oracle" means here -- bypassing Stage
B's blur->dynamics prediction). So any usable test dataset must supply the sharp high-fps sequence, not
merely a blurred frame. Checked the actual input contract
(`checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageA_burst3_ar.yml` + `dataset/adobe240_dataset.py`):
`burst_size=3`, `window_size=11` (an 11-frame average makes each blur image), blur "centers" are the
scene's listed frames (Adobe240's standard ~8-frame center spacing), so one burst needs roughly 27
contiguous native-fps sharp frames per test clip (2x8 between the 3 centers, +/-5 half-window padding on
each end), and `wan_num_frames=25` real frames are pulled from the centered sub-window of that span as
the GT video for comparison, at 1280x736 (hires). That rules out:
- **Physics-IQ** (`github.com/google-deepmind/physics-IQ-benchmark`, DeepMind, 396 real filmed videos
  purpose-built to test generative video models' physical understanding, 3840x2160 @ **30fps**) --
  disqualified *for this oracle test specifically*: 30fps gives no sub-frames to average into a
  training-style blur while also retaining sharp GT frames at the needed density. Worth revisiting for
  the deployable Stage C test later (blur-only input), just not this one.
- **Morpheus** (arXiv 2504.02918, 130 real-world Newtonian-mechanics videos) -- same category/purpose as
  Physics-IQ; fps/resolution not confirmed but likely the same standard-video-fps issue. Same verdict:
  candidate for Stage C, not oracle.

**FMO / TbD / TbD-3D / Falling Objects datasets** (Rozumnyi et al., CTU Prague,
`cmp.felk.cvut.cz/fmo/`) survive the check -- the strongest technical match, because each sequence
already ships a real high-speed ground-truth video **plus** a temporally-averaged low-speed/blurred
version, i.e. built around exactly the same "average N high-fps frames -> one blurred frame, keep the
sharp burst as GT" methodology this project's own training pipeline uses. Falling Objects Dataset =
`falling_imgs_gt.zip` (dropped box/cell/key/rubber etc., high-speed + low-speed versions, GT in text).
FMOv2 = sports balls in motion (volleyball, tennis, darts, frisbee, ping pong, squash, hockey). TbD /
TbD-3D = striking golf/tennis/volleyball/badminton balls, a falling cube and coin, 240fps source with GT
trajectories, PNG or MATLAB format. All free direct-download zips, no account/license gate seen. **Not
yet verified**: whether individual TbD/FMO clips are long enough to supply the ~27-contiguous-frame span
above -- these are often short (an object crossing the frame once), so a per-sequence frame-count check
is needed before committing to this route.

**Fastest actual route, per advisor consultation:** rather than downloading anything new, GoPro (240fps,
already on depot at `/depot/chan129/users/harshana/GoPro_flat`) and Adobe240 (the training source itself,
held-out scenes) both already satisfy the frame-span/fps/resolution contract with zero new dataset code.
Hand-picking a few held-out clips with recognizable physical motion (something falling, thrown, bouncing,
swinging) and running the existing bidirectional hires eval path unchanged gives a real
generalization/upper-bound datapoint today. FMO/TbD becomes the follow-up once/if that's not enough
variety of "known" motions. Also surfaced but lower priority: **Bounce dataset** (5K RGB-D
bouncing-foam-ball trajectories -- depth-focused, not an RGB-video-quality target) and **PhyWorldBench**
(arXiv 2507.13428, another physical-realism T2V benchmark).

**Other high-fps datasets surveyed (for future training/eval beyond Adobe240fps, which is already the
training set):**
- **X4K1000FPS** (Sim et al., XVFI, ICCV 2021) -- 4K @ 1000fps, 175 scenes x 5000 frames, Phantom Flex4K
  camera. Highest fps/resolution option found.
- **BVI-HFR** / **BVI-DVC-4K** (Bristol Vision Institute) -- high-frame-rate 4K sequences (up to ~1000fps
  variants exist in the BVI family), used for HFR video coding/VFI research.
- **GOPRO_Large** -- 240fps, same class as Adobe240fps, already familiar to this codebase from the
  deblurring baselines (FFTformer/UFPNet GoPro runs, see [[gopro_baselines_unique_scenes]]).
- **SportsSloMo** (arXiv 2308.16876) -- human-centric high-fps sports benchmark, could be a source of
  additional "known motion" clips (throwing, jumping) beyond simple ball physics.
- **Need for Speed (NfS)** -- 240fps real-world tracking dataset, diverse object motions including some
  ball/sports content.

No experiments run yet this session beyond the status check, backup, and this survey -- next step is
picking a dataset (FMO/TbD look like the best fit given the blur-averaging methodology match) and
actually running Stage A oracle inference on a handful of clips.

## First real oracle generalization test: Falling Objects (FMO/TbD), reversal rate reverts to 50/50 (2026-09-21)

Downloaded the CTU FMO/TbD "Falling Objects" dataset (`falling_imgs_gt.zip`,
`cmp.felk.cvut.cz/fmo/`) to `/scratch/gilbreth/wweligam/data/FMO_TbD/falling_objects/` -- 6 scenes
(box/cell/key/marker/pen/rubber dropped from a table), 500 contiguous real 240fps sharp frames per
scene under `imgs_gt/<scene>/`. Wrote `dataset/falling_objects_dataset.py` to adapt this to the oracle
model's exact burst/blur-averaging input contract: synthesizes each segment's blur image by averaging
`window_size` (11) consecutive sharp frames on the fly (no pre-made blur file needed, unlike
Adobe240Dataset), builds non-overlapping `burst_size`-chunks of centers `center_stride` (8) apart
(mirroring Adobe240's own center spacing), and -- since native frames are portrait 300x960 (aspect
~0.31, far outside the trained `aspect_ratio_range: [0.5, 2.0]`) -- pillarbox-pads each frame to square
before resizing to `out_size` (736), preserving the full vertical fall trajectory instead of squishing
or cropping it away.

Ran oracle inference (`Wan22VJEPADynamics_stageA_burst3_ar_infer_fallingobjects.yml`, checkpoint-140000,
`dynamics_source: vjepa_gt`) on 18 bursts (3 evenly-spaced-in-time per scene x 6 scenes) on the idle
gilbreth-k024 A100 -- completed cleanly in ~10.5 min, no errors.

**First read was misleading on two counts, both now corrected:**
1. The in-loop comet `video_psnr`=32-37dB and a first bidirectional-metrics pass on the raw
   (uncropped) videos gave PSNR=76.83dB -- absurdly high. Root cause: pillarbox padding covers ~65% of
   each 736x736 frame in pure black, and black-vs-black trivially inflates PSNR/SSIM across most of the
   frame regardless of how good the actual content region is. Re-ran
   `analysis/compute_video_metrics.py` on padding-cropped copies of the same videos (content region
   verified empirically at columns 240:496 of 736, via nonzero-pixel column range on a real frame) --
   corrected numbers: **PSNR=36.62, SSIM=0.9208, LPIPS=0.0117, EPE=5.220, FVD=186.98 (n=18)**. Still
   above the 29.94dB in-distribution Adobe240 hires number, but not a fair "beats in-distribution"
   comparison -- see point 2.
2. Visual inspection (frames extracted from `wan_output_`/`gt_` videos) showed several sampled bursts
   landed on mostly-static pre-drop or post-landing stretches rather than the actual fall -- confirmed
   quantitatively via frame-to-frame pixel-diff motion detection per scene (e.g. v_box_GTgamma's peak
   motion is at frame ~301 of 500; the evenly-spaced sampling scheme's nearest burst, centered at 241,
   sits just before that window). A mostly-static reconstruction target is an easy case that inflates
   PSNR/SSIM/LPIPS without testing whether the model gets the actual DYNAMICS right -- the corrected
   36.62dB number above is real but should be read as "the model reconstructs static/near-static
   real-world scenes well," not as a dynamics-generalization result.

**The one number that survived both corrections, and is the actual finding: direction reversal is
9/18 = 50% -- a dead coinflip.** This holds on both the raw and padding-cropped scoring (same
per-clip forward/reversed calls either way, cropping only changed PSNR/SSIM/LPIPS/EPE magnitudes, not
direction). This is a stark contrast with the Adobe240 hires oracle eval, which has held a clean 0/8
reversed for the last four checkpoints of training (see the 2026-09-16 through -18 entries above) --
i.e. burst-mode's core fix (a temporally-ordered, non-time-symmetric input should make the correct
generation direction recoverable) **does not generalize off-distribution**. Plausible explanations,
not yet distinguished: (a) the model learned to exploit some Adobe240-specific shortcut for direction
(motion-blur trail statistics, scene-type priors, etc.) rather than genuinely reading segment order from
the burst; (b) the extreme portrait-canvas pillarbox-padding preprocessing (novel to this data, never
seen in training) disrupts whatever positional/token-order signal burst-mode relies on; (c) some
Falling-Objects-specific quirk (near-static windows, in particular) removes the very temporal asymmetry
burst-mode depends on to disambiguate direction in the first place. (b) and (c) are testable by
re-running on real in-distribution-aspect-ratio content with recognizable dynamics (e.g. GoPro/Adobe240
held-out clips at the native ~1.7:1 aspect, per the advisor's original "fastest route" suggestion) and
by deliberately targeting each scene's real high-motion window (in progress, see below) instead of a
mix of static and moving windows.

**Follow-up in progress**: re-running on exactly one burst per scene, deliberately centered on each
scene's peak-motion frame (via the same frame-diff method) instead of evenly-spaced-in-time sampling --
`Wan22VJEPADynamics_stageA_burst3_ar_infer_fallingobjects_peak.yml`, launched on gilbreth-k026. Peak
centers used: box=301, cell=156, key=270, marker=313, pen=341, rubber=304 (out of each scene's 500
frames). This isolates the "does it generate the right/complete fall" question from the "is it even
looking at a dynamic window" confound above, and gives a second, independent reversal-rate datapoint
that (b)/(c) above predict differently from (a): if (b) or (c), the reversal rate should improve once
bursts are centered on real motion; if (a), it likely won't.

**CRITICAL codebase gotcha found while debugging the peak-motion launch's first crash**:
`utils/misc.py::DictAsMember` (the plain-dict-free config wrapper `trainer.py` builds `opt` from, via
`DictAsMember(**data)` after the OmegaConf->yaml->dict round-trip) subclasses `dict` but stores its
actual contents in `self.__dict__`, NOT the real underlying dict storage -- so `dict(some_dictasmember)`,
`key in some_dictasmember`, and `some_dictasmember[key]` all silently operate on an EMPTY dict (no
error, just always "not found"), while `.get(key, default)` is the one method DictAsMember correctly
overrides to actually read `self.__dict__`. `dataset/falling_objects_dataset.py`'s first version did
`self.peak_center_scenes = dict(peak_centers)` then `scene not in self.peak_center_scenes` --
`dict(peak_centers)` silently produced `{}` (CPython's dict() constructor fast-paths off the real
dict storage of a dict subclass instance rather than calling `.items()`), so every scene "wasn't found"
and `__init__` raised `ValueError("No valid bursts constructed")` on the peak-motion job's first launch
attempt. Fixed by keeping the raw opt value and only ever calling `.get()` on it (never `dict()`, `in`,
or `[key]`) -- verified against a real `DictAsMember` instance (not just OmegaConf, which doesn't have
this bug) before relaunching. **Any new dataset/model code that reads a nested-dict config value must
use `.get()` exclusively, never `dict(...)`, `in`, or `[...]`, on whatever `opt` object it's handed** --
this silently produces empty results with zero error signal, exactly the kind of bug this project's
other "silent failure" lessons ([[peft_resume_optimizer_disconnect]], [[accelerate_save_optimizer_noop]])
warn about.

## Peak-motion rerun: the model doesn't fail to disambiguate direction, it fails to render the object at all (2026-09-21)

With the `DictAsMember` bug fixed, `Wan22VJEPADynamics_stageA_burst3_ar_infer_fallingobjects_peak.yml`
ran cleanly: 6 samples, one per scene, each burst deliberately centered on that scene's real
frame-diff-detected motion peak (box=293, cell=148, key=262, marker=305, pen=333, rubber=296 --
adjusted slightly from the earlier peak-detection numbers to keep centers inside the valid
margin-bounded range). Content-crop bidirectional metrics:
**PSNR=36.02, SSIM=0.9144, LPIPS=0.0140, EPE=5.401, FVD=1175 (n=6; FVD unreliable at n=6, ignore it)**.
Direction flipped to **6/6 forward, 0/6 reversed** (vs. the evenly-spaced pass's 9/18=50%) -- see below
for why this should NOT be read as "burst mode's direction disambiguation works fine off-distribution
after all."

**The real finding, from visual inspection (not the aggregate metrics):** extracted a dense
every-3rd-frame strip (GT top row, prediction bottom row) for `v_pen_GTgamma_c333_burst3` -- saved at
`/tmp/.../scratchpad/peak_frames/pen_full_strip_small.png` during this session, worth regenerating if
revisiting. GT clearly shows the pen (a thin white object) falling diagonally past the chair leg across
frames, landing near the bottom by frame ~21. **The prediction shows this object in ZERO of the 9
sampled frames spanning the entire clip** -- the scene looks essentially frozen (hand, table, chairs,
background all static and matching the input frame) with no falling object rendered anywhere,
anytime. Spot-checked `v_box` and `v_rubber` individually (full detail) plus `v_cell`/`v_key`/`v_marker`/
`v_pen` via a mid-clip composite grid -- **all 6/6 scenes show the same pattern**: static background
faithfully reproduced, the actual falling object never rendered.

This directly explains the suspiciously-high content-crop PSNR/SSIM in BOTH passes (29-37dB range,
above even the 29.94dB in-distribution Adobe240 number): the falling object occupies a tiny fraction of
each frame's pixels, so a prediction that reproduces the static background/foreground (hand, furniture)
correctly while completely omitting the small moving object still scores very well on
whole-frame PSNR/SSIM/LPIPS -- **these metrics are not sensitive to whether the actual dynamic content
of interest was generated at all** in this kind of scene. This is a methodological trap specific to
small-object-motion content and worth remembering for any future oracle/generalization test on this
kind of data: high PSNR/SSIM here is NOT evidence of successful video generation.

It also reframes the reversal-rate result from both passes. A frame-to-frame temporal-variance check
(mean abs pixel diff between consecutive frames) actually showed PRED variance 1.5x-5.6x HIGHER than
GT's in every one of the 6 peak-motion scenes -- at first read this looked like it contradicted the
"prediction is static" finding, but the dense frame strip resolves it: the extra pixel-level variance is
diffusion temporal flicker/noise (color/texture jitter across frames), not coherent object motion -- the
composition stays static while individual pixels jitter frame to frame. Given the prediction never
depicts the actual falling object in the first place, the bidirectional scorer's forward-vs-reversed
pick (which only compares whole-clip PSNR in each direction) is being decided by this flicker structure
and/or the deblur-prefix-anchored first frame, not by any genuine reading of which way the (absent)
object's motion goes -- so **neither the evenly-spaced pass's 50/50 split nor the peak-motion pass's
6/6-forward should be read as evidence about direction disambiguation one way or the other**. The
premise the reversal metric depends on (the model is actually generating the scene's real dynamics) is
false here, so the metric itself isn't measuring what it's designed to measure on this data.

## Conclusion: what "oracle Stage A upper bound" actually looks like on real, out-of-distribution motion (2026-09-21)

Answering the session's original question directly: on real footage of a genuinely recognizable known
motion (an object falling off a table, real 240fps camera, oracle ground-truth V-JEPA dynamics from the
real future frames), **the trained Stage A oracle checkpoint-140000 does not generate the correct
video** -- it reproduces the static parts of the scene faithfully but drops the actual falling-object
motion entirely, across all 6 tested scenes (box, cell, key, marker, pen, rubber). This is a real
failure to generalize beyond the Adobe240 training distribution for objects with a small pixel
footprint and fast, large-displacement motion (falling under gravity, not the panning
camera-shake-style blur Adobe240 is built around) -- notably different from, and more fundamental than,
the burst-mode reversal-rate problem the project's prior work focused on: here the model isn't getting
the direction of a generated motion wrong, it's not generating the motion's actual subject at all.
Caveats on generalizing this conclusion further: n=6 scenes from one dataset, all sharing the same
capture setup (fixed camera, indoor, small dropped object) and the same out-of-distribution pillarbox
preprocessing this session introduced -- worth checking against in-distribution-aspect-ratio real motion
(e.g. GoPro/Adobe240 held-out clips, the advisor's original "fastest route" suggestion, never yet run)
to separate "doesn't generalize to this content style" from "doesn't generalize to this canvas shape."

## Square-crop (no-padding) high-res rerun: ruled OUT the aspect-ratio/padding explanation (2026-09-21)

The padded-canvas caveat above is now tested and closed. Added `crop_mode: motion_square` to
`dataset/falling_objects_dataset.py`: instead of pillarbox-padding the native 300x960 portrait frame
into a square that is ~65% black, it takes a genuine **300x300 native crop with zero padding**,
positioned by a real frame-diff motion-energy scan of each burst's own span
(`_motion_square_offset`, picks the `side`-tall window maximizing captured motion energy), then
upsamples that crop to `out_size=736`. This is a true 1:1 canvas (matching Adobe240Dataset's own native
`crop_size: 320` square convention), full-frame real content, and the object gets a much larger
relative pixel footprint. Verified before launching that the crop tracks the fall in all 6 scenes
(object clearly visible entering top and exiting bottom of the crop). Reran the identical peak-motion
centers through checkpoint-140000, hires.

**These three passes are NOT mutually comparable -- do not read them as a progression.** Each row is a
within-pass record only. The padded passes scored a 256x736 band of the full-height frame; the
motion_square pass scored a 300x300 native region upsampled 2.45x. Different spatial extent, different
content, different upsampling factor -- so a PSNR "delta" between rows is not a measurement of
anything, and no delta is quoted here deliberately.

```
Pass                                     canvas                     PSNR    SSIM   LPIPS     EPE   reversed   n
padded, evenly-spaced (content-crop)     pillarbox 736^2, 65% black 36.62  0.9208  0.0117   5.220    9/18     18
padded, peak-motion (content-crop)       pillarbox 736^2, 65% black 36.02  0.9144  0.0140   5.401    0/6       6
motion_square, peak-motion (raw)         true 300^2 crop -> 736^2   40.22  0.9650  0.0106   0.640    3/6       6
```

Two reasons the motion_square row's higher PSNR/SSIM and near-zero EPE are not evidence of better
generation, beyond the incomparability above: the square crop is a much more heavily upsampled,
smoother, lower-spatial-frequency reconstruction target (easier to reproduce), and EPE sits near 0
because both GT and prediction have near-zero optical flow almost everywhere once the frame is
dominated by static background -- the object's large displacement occupies too few pixels to move a
whole-frame mean. Direction is back to 3/6 reversed, i.e. the coinflip, confirming the earlier peak
pass's 6/6-forward was not a real signal.

**Controlled test of the only question that matters -- does the model render the falling object?**
Built a per-frame "moving-content map" (each frame's abs deviation from the clip's temporal median,
which removes the static background and leaves the moving object), then measured the mean per-frame
spatial correlation between GT's and the prediction's moving-content maps. Crucially this was run with
BOTH a negative control (GT vs time-reversed GT -- a deliberately wrong pairing) and a positive control
(GT vs GT shifted 2 frames -- an "almost correct" rendering with a small timing error), so the metric's
own sensitivity is measured rather than assumed:

```
scene                      pos ctrl (GT vs GT+2f)   PREDICTION   neg ctrl (GT vs reversed GT)
v_box_GTgamma                       0.250            -0.024               0.022
v_cell_GTgamma                      0.147             0.000               0.053
v_key_GTgamma                       0.567             0.034               0.134
v_marker_GTgamma                    0.681             0.100               0.117
v_pen_GTgamma                       0.068             0.055               0.035
v_rubber_GTgamma                    0.107             0.003               0.049
```

In the 4 scenes where the metric demonstrably has discriminative power (box, key, marker, rubber --
positive control clearly exceeds negative control), **the prediction scores at or BELOW the negative
control**: its frame-to-frame content has no more spatial agreement with the real falling object than a
deliberately time-reversed mismatch does. For pen and cell the positive control is itself weak (the
object moves far enough in 2 frames that even a near-correct rendering decorrelates), so those two are
inconclusive rather than passing. Separately, total moving-content energy in the prediction is
comparable to GT's (0.23x-1.53x) -- so the model is NOT producing a frozen video; it produces plenty of
frame-to-frame change, just none of it corresponding to the object. That reconciles the earlier
padded-pass puzzle where predicted temporal variance measured 1.5x-5.6x HIGHER than GT's.

**This also corrects an error in this session's own earlier visual read.** From every-3rd-frame
thumbnail strips I reported that `v_pen` showed the model "genuinely tracking the falling pen's
diagonal trajectory" and that cell/key/marker showed a "static, non-moving smudge." Both readings are
wrong: pen's faint diagonal streak scores 0.055 vs a 0.035 negative control (chance), and
cell/key/marker are not static at all (they carry real moving-content energy, just misplaced). Faint
low-contrast streaks in downscaled thumbnail strips are not reliable evidence -- the controlled
correlation test is. Any future qualitative claim about "did it render the motion" on this data should
go through the control-flanked measurement, not eyeballing.

**Ruled out: degenerate oracle conditioning.** Because the padded pass handed V-JEPA a square canvas
that was mostly zeros while the motion_square pass handed it a true square (which
`vjepa_from_unit_tensor` passes through as a documented byte-identical no-op, skipping letterboxing),
it was worth confirming the oracle tokens in this run actually carry content. Inspected the
dynamics-token PCA panel that `validate_step` logs as the top-right quadrant of
`wan_pred_vs_gt_*.mp4`: rich spatial structure clearly tracking real scene geometry (chair legs,
floor/background separation), per-frame spatial std 54.7/255, temporal std at fixed pixels 18.6/255,
36211 unique colors in frame 0. The oracle conditioning is non-degenerate, so "the model was handed
empty tokens" is not the explanation -- the failure is in the generation path.

**Revised conclusion:** the padded canvas / extreme aspect ratio was NOT the explanation. Giving the
oracle model a clean square, no-padding, high-resolution crop in which the object is large and clearly
visible changes the aggregate metrics substantially but does not make it render the falling object,
and the oracle tokens it was conditioned on were verified non-degenerate. Scope this claim carefully:
what is established is that on this content, two canvas treatments (pillarbox-padded full frame, and
native square motion-tracked crop) both fail, so canvas shape/padding does not explain the failure --
NOT that the model fails under any conceivable canvas. n=6 scenes, one dataset, one capture setup, and
in 2 of those 6 (pen, cell) the positive control was too weak to discriminate. Still open, and now the
single most informative next test: the same controlled measurement on in-distribution content
(GoPro/Adobe240 held-out clips at native aspect), to establish whether the model renders motion
faithfully even *in* distribution or whether this failure mode was always present and simply invisible
to PSNR-style metrics.

## WHY it fails OOD: in-distribution "motion" is global camera shake, OOD motion is a small local object (2026-09-21)

Ran the identical control-flanked measurement on the burst-3 model's OWN in-distribution Adobe240
validation outputs (the 8 held-out scenes already written to
`Wan22VJEPADynamics_stageA_burst3_ar_2026_09_16_15_03_14/videos/` at checkpoint-140000 -- free, no GPU
needed):

```
scene                   pos ctrl(+2f)   PREDICTION   neg ctrl(rev)
720p_240fps_1_00017            0.751        0.573          0.270
GOPR9635_00017                 0.762        0.447          0.467
GOPR9637a_00017                0.788        0.639          0.525
IMG_0004a_00017                0.730        0.488          0.271
IMG_0015_00017                 0.784        0.559          0.745
IMG_0023_00017                 0.838        0.431          0.501
IMG_0179_00017                 0.824        0.713          0.398
IMG_0183_00017                 0.801        0.693          0.312
MEAN                           0.785        0.568          0.436
```

**In-distribution the model genuinely does render motion**: prediction 0.568 mean, clearly above the
0.436 negative-control floor and below the 0.785 positive control -- i.e. real but imperfect, roughly
midway between "wrong" and "almost right" (5/8 scenes clearly above their own negative control, 3/8 at
or below). Compare the OOD falling-objects numbers in the section above: prediction 0.00-0.10 against
a 0.02-0.13 floor, i.e. at chance. So the failure is specifically an out-of-distribution failure, NOT
a latent defect that was always there and hidden by PSNR. The project's in-distribution numbers are
measuring something real.

**The mechanism, and it is structural.** Look at the negative control itself across the two datasets:
0.436 in-distribution vs 0.02-0.13 on falling objects. That gap is the whole story. Adobe240 blur is
*camera shake*: the entire frame moves coherently, so the "moving-content" map is spatially broad and
looks much the same whichever temporal direction you read it -- hence a time-reversed mismatch still
scores 0.436. Falling-objects blur is a *small rigid object translating against a perfectly static
background*: the moving-content map is a tiny localized blob whose position is the entire signal, so a
mismatch scores ~0.05. These are structurally different tasks:

- In-distribution, generating "the motion" means globally warping/animating a frame whose content is
  entirely present in the blurred conditioning image. Nothing has to be invented or localized.
- OOD, generating "the motion" means placing a small object at the right position in each frame
  against a background that must stay still. That is a localization/synthesis problem the training
  distribution never poses.

This also rules out the two explanations that were still open. The blur input is NOT devoid of the
object -- measured peak object contrast retained through the 11-frame averaging is 18-52% of the sharp
frame's, and 1.2x-21x above the blur-vs-background noise floor depending on scene (box is the
marginal one at 1.24x, key the strongest at 20.9x). And the oracle V-JEPA tokens demonstrably carry
the motion (PCA panel shows clear structure; the user independently confirmed the falling object is
visible in it). So the object's location and trajectory are present in BOTH conditioning pathways, and
the model still does not render it -- consistent with the model having learned to use the dynamics
tokens as a global frame-warping signal (which is all Adobe240 ever required of them) rather than as a
"put this object here" localization signal.

Note for anyone re-running this comparison: the in-distribution videos are 320x320 native
training-time validation outputs while the falling-objects ones are 736x736, so the raw correlation
magnitudes are not comparable across the two tables. Each table is only interpretable against its own
positive/negative controls, which is exactly how it is read above.

## Single-blur-image (burst_size=1) control: fails identically -- not a burst-extension defect (2026-09-21)

Per user request, ran the **single-frame** line of work on the same data as a direct control. Used the
single-frame current-best recipe (findings' recipe-progression step 5): LoRA + `deblur_prefix_frames=4`
+ `aspect_ratio_range` training, pooled connector, checkpoint-60000 from
`Wan22VJEPADynamics_stageA_lora_deblur_ar_2026_09_07_03_38_44` -- which scored **31.56dB**
in-distribution, i.e. BETTER than the burst-3 model's 29.94dB, so this is not a weaker model.
Config: `checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageA_singleframe_infer_fallingobjects.yml`.
Everything else held identical to the burst run: same 6 scenes, same peak-motion centers, same
no-padding `motion_square` crop, same `out_size: 736`. `dataset/falling_objects_dataset.py` gained a
`burst_size == 1` branch emitting the single-frame contract the model's non-burst path indexes
(hardcoded `blur` + `wan_frames` keys; its vjepa_gt oracle branch reads the V-JEPA clip straight out
of `wan_frames`), mirroring `Adobe240Dataset._finalize_sample`. `wan_num_frames` must stay 9 (what
this checkpoint trained with), so the generated clip covers 9 frames rather than the burst model's 25.

```
scene                      pos ctrl (GT vs GT+2f)   PREDICTION   neg ctrl (GT vs reversed GT)
v_box_GTgamma                       0.056             0.013              0.106      (no power)
v_cell_GTgamma                      0.065             0.030              0.142      (no power)
v_key_GTgamma                       0.463             0.019              0.357
v_marker_GTgamma                    0.589             0.323              0.391
v_pen_GTgamma                       0.035            -0.006              0.110      (no power)
v_rubber_GTgamma                    0.107             0.050              0.143      (no power)
MEAN                                0.219             0.072              0.208
```

**Same failure.** In the only 2 scenes where the metric has discriminative power (key, marker --
positive control exceeds negative), the prediction scores below the negative control, exactly as the
burst-3 model did. The other 4 scenes are inconclusive: their positive control falls *below* their
negative control, i.e. the metric has no power there at all. That loss of power relative to the burst
run is expected and is a `wan_num_frames` artifact -- at 9 frames a 2-frame shift disrupts 2/9 of the
clip versus 2/25 for the burst model, so the positive control is much harsher here. Do not read the
lower absolute numbers in this table as "worse than the burst model"; the two tables have different
control scales and are not directly comparable.

**Conclusion: the failure is NOT something the burst extension introduced.** Both the 3-image burst
model and the single-blur-image model, both at their respective best checkpoints, both given oracle
ground-truth V-JEPA dynamics, fail to render a small falling object on this data.

**The single-frame run also produced the most interpretable picture of the failure mode yet** (visual,
`v_marker`, the highest-scoring scene -- GT top row, prediction bottom row, all 9 frames): GT shows the
marker falling diagonally down-right across the clip. The prediction shows a **pale vertical smear that
stays in essentially the same place in every frame** -- the object's motion-blur *trail* rendered as a
static streak, rather than a discrete object traversing that trail over time. That is a precise
statement of what the model does wrong, and it fits the in-distribution mechanism above exactly: on
Adobe240, "deblur the frame" and "animate the frame" are the same global operation, so the model never
had to learn the distinct skill of converting a localized blur streak into a sharp object that moves
along it. Here it reproduces the streak -- faithfully, in the right place -- and stops there.

## Fine-tune launched: FMO/TbD + Adobe240 mixed, to add localized-object motion (2026-09-21)

Acting on the diagnosis above (the model learned dynamics tokens as a global frame-warping signal
because that is all Adobe240's camera-shake blur ever required), launched a fine-tune that mixes the
missing regime into the original distribution rather than replacing it.
Config: `checkpoints/MotionBlur2Video/Wan22VJEPADynamics_burst3_finetune_fmo_mixed.yml`, resuming
burst-3 `checkpoint-140000`. New code: `dataset/mixed_dataset.py` (`MixedDataset`, fixed-ratio
multi-source sampling with a deterministic per-epoch plan) plus `random_square` crop mode,
`centers_around_peak` and `burst_center_step` in `dataset/falling_objects_dataset.py`.

**Extracted TbD-3D** (`/scratch/gilbreth/wweligam/data/FMO_TbD/tbd3d/`) -- 10 sequences, 300-650
contiguous frames each, 960x400 landscape, same `imgs_gt/<scene>/NNNNNNNN.png` layout as Falling
Objects so `FallingObjectsDataset` reads it unchanged. Without it there were only 6 scenes total and
no honest held-out set.

**Split chosen by measurability, computed BEFORE picking it** (pure CPU, from GT frames alone:
positive control = GT vs GT+2frames, negative = GT vs time-reversed GT, margin = pos-neg):

```
FO  v_marker 0.551   FO  v_key  0.396   FO  v_box   0.218   FO  v_cell  0.059(weak)
FO  v_pen    0.003(NO)  FO v_rubber 0.043(NO)
TbD outf1 0.598  outb1 0.490  depthf2 0.480  out2 0.428  depthf3 0.358  out1 0.358
TbD depthf1 0.349  depthb2 0.312   (depth2, outa1: peak too near clip start for a full burst span)
```

HELD OUT: v_key, v_marker, HighFPS_GT_out2, HighFPS_GT_outf1 (2 FMO + 2 TbD, all strong margins).
TRAIN: the other 12, deliberately including the unmeasurable ones (v_pen/v_rubber/v_cell) -- low
metric power is not low data quality. Leakage verified zero. **This corrected a planned split of
v_pen + v_rubber as held-out, which would have been unprovable either way** (margins 0.003 / 0.043) --
the advisor caught it; the lesson is to establish eval power before choosing what to hold out.

**Three deliberate choices, each against the naive option:**
- *Random, not motion-centred, training crops* (`crop_mode: random_square`). motion_square puts a big
  moving object dead centre every sample; training only on those teaches "a crop from this source has
  an object in the middle", a spurious correlate Adobe240's random crops do not share. Random
  positioning also multiplies distinct samples from only 12 clips. motion_square stays for EVAL.
- *25% new data, not 50%.* 12 fixed-camera clips memorize easily. If held-out never moves while
  train-scene reconstruction sharpens, that is memorization and a higher ratio makes it worse. Raise
  only on evidence.
- *AR augmentation OFF.* Mixed batches need one fixed square shape (MixedDataset uses default_collate,
  not Adobe240Dataset's AR-deferred collate_fn). Tradeoff: AR was recipe-step-5's win at 1280x736, so
  the hires eval must be checked at the FIRST checkpoint, not only at the end.

**LR bug caught before launch, would have been silent.** checkpoint-140000 stores `initial_lr=1e-4`
and `base_lrs=[1e-4]` (it only reached 2e-5 via its cosine_floor schedule; verified by loading its
`optimizer.bin`/`scheduler.bin` directly). A fresh `constant` schedule recomputes
`lr = initial_lr * lambda`, so the fine-tune would have run at **1e-4, 5x the intended rate** -- the
exact catastrophic-forgetting risk this mix is designed to avoid, with no warning. Added
`Wan22VJEPADynamics_model.after_resume_checkpoint` (the documented unwrap pattern: AcceleratedOptimizer
proxies `.param_groups` live; AcceleratedScheduler hides `base_lrs`/`_last_lr` one level down at
`.scheduler`), gated by `train.optim.override_lr_on_resume`. Smoke-confirmed:
`forced resumed LR to 2e-05` and `lr_0=2e-5` held across steps.

Smoke test also caught a `NameError` (this model takes `logger` as an `__init__` arg; other methods
must use `self.logger`) -- worth remembering before adding any new method that logs here.

Run: 140000 -> 150000 steps, batch 2, ckpt every 1000, val every 500 on the 4 held-out scenes.
NOTE gilbreth-k024 is concurrently running the user's own
`VAEBlurAlignFluxLoRA_GoPro_noPretrain_seed1234` (68GB) -- this fine-tune is on k026. An earlier smoke
attempt OOM'd on k024 purely from that co-tenancy, not from any fault in this config.

## deblur_prefix_frames divisibility guard enforces the WRONG congruence (2026-09-21)

Found while working out the temporal layout for a proposed n-blur-images x r-repeats conditioning
scheme. Wan's VAE is **causal**: `t_lat = (T-1)//4 + 1`, so latent 0 absorbs exactly **one** pixel
frame and every later latent absorbs 4. `patch_size=(1,2,2)` (p_t=1), so post-patch temporal positions
== latent frames.

Consequence: a prefix of p pixel frames lands on a latent boundary only when **p = 4k-3**, i.e.
p in {1, 5, 9, 13, ...} -- p ≡ 1 (mod 4). Verified by explicit pixel->latent mapping:

```
p= 1 -> covers latents [0]      clean      p%4=1
p= 4 -> covers latents [0,1]    STRADDLES  p%4=0   <- what the guard requires
p= 5 -> covers latents [0,1]    clean      p%4=1
p= 8 -> covers latents [0,1,2]  STRADDLES  p%4=0
```

`Wan22VJEPADynamics_model.__init__` raises unless `deblur_prefix_frames % vae_scale_factor_temporal
== 0` -- so it **accepts exactly the values that straddle and rejects exactly the values that align**.
The guard's comment says it exists to make "the extra pixel-frames map to a whole number of extra
latent frames", which is the right intent applied with the wrong arithmetic (it assumes a flat T/4
divide and forgets the causal 1-frame first slot).

Live impact on every config in this line (all use `deblur_prefix_frames: 4`, T=25+4=29, t_lat=8):
latent 0 <- pixel 0 (prefix), latent 1 <- pixels 1-4 = **3 prefix pixels + 1 real frame**. Meanwhile
`_deblur_prefix_post_patch_frames()` returns `4//4//1 = 1`, so `_dynamics_tokens_with_deblur_padding`
zero-pads the dynamics tokens over latent 0 only -- latent 1, which is 3/4 static prefix content,
still receives real motion tokens. Small but systematic: motion signal injected onto mostly-static
padding.

Not fixed yet -- changing it shifts the latent layout and would invalidate comparability with
checkpoint-140000 and every number in this file. Fix it only alongside a deliberate re-train, and
prefer p=5 over p=1 if the current T=29 sequence length should be preserved (p=5 -> t_lat=8, 2 clean
padding latents + 6 content latents).

**Clean layouts for an n x r conditioning scheme** (input n*r + p frames, target = GT video with its
own first frame repeated p times -- which is what `_prepend_deblur_frames` already builds). Requires
n*r = N so both sides align:

| p | n | r | N=n*r | T  | t_lat | layout                  |
|---|---|---|-------|----|-------|-------------------------|
| 1 | 3 | 8 | 24    | 25 | 7     | 1 pad + 3x2 content     |
| 5 | 3 | 8 | 24    | 29 | 8     | 2 pad + 3x2 content     |

Both satisfy the dataset's own `(wan_num_frames-1) % 4 == 0` guard. The current config (p=4, N=25) is
in neither family: p=4 straddles, and 25 does not split into 3 equal chunks.

## Implemented: burst-blur conditioning -- all 3 segments into Wan, n x r temporal layout (2026-09-21)

Per user request. **Stage B already did this**: `_dynamics_full_grid_burst`'s `blur_jepa` branch
already loops `for s in range(burst_size): blur_image = burst_blur[:, s]` and concatenates the
per-segment predictor grids, so V-JEPA has always seen all 3 blur images. Only Stage A fed a single
image, via `_first_frame_condition` -> `burst_blur[:, 0]`.

**Why the obvious route is wrong.** The natural implementation -- extend `first_frame_mask` to cover
the blur frames -- is a dead end, and the loss shows why:
`loss = (per_elem * frame_mask).sum() / frame_mask.sum()`. A masked frame is BOTH hardwired to the
condition AND excluded from the loss. Filling 6 of 8 latents with blur would make the output video
literally be the 3 blur images and delete most of the training signal. Confirmed by reading the loss,
not assumed.

**What was built.** Conditioning is assembled in PIXEL space exactly as specified (each segment
repeated over the r frames it was averaged from, preceded by p prefix repeats of segment 1), encoded
once, and added to the hidden states entering block 0:

```
p=5, n=3, r=8 -> 29 pixel frames -> t_lat 8   (identical token count to the control arm)
  latents 0-1 <- prefix(blur1)   2-3 <- blur1   4-5 <- blur2   6-7 <- blur3
  verified: no latent mixes two segments
```

Injected through a new zero-init `WanBlurCondEmbed` (Conv3d, kernel==stride==patch_size, z_dim 48 ->
inner 3072). Adding at block 0 is mathematically identical to widening `patch_embedding`'s input
channels and concatenating, because that conv is linear in its channel dim -- but the weights live in
a separately-registered module appended to `self.models`, so they are actually saved. Widening
`patch_embedding` would have buried new trainable weights inside the PeftModel, which saves adapter
tensors only (`apply_lora_to_model` sets no `modules_to_save`) -- they would have trained and been
silently discarded at every checkpoint, the [[module_swap_checkpoint_compat]] failure shape.

**Guard fixes shipped alongside** (both were wrong in the same way -- assuming a flat T/4 divide and
forgetting the causal VAE's 1-pixel first latent):
- `deblur_prefix_frames` guard now accepts `p % 4 == 1` (latent-aligned, preferred) and still accepts
  `p % 4 == 0` purely so every existing p=4 checkpoint/config keeps loading.
  `_deblur_prefix_post_patch_frames` computes the correct count for the aligned family and preserves
  the historical under-count for the legacy one.
- `wan_num_frames` divisibility guard in `adobe240_dataset.py`, `x4k1000fps_dataset.py` and
  `falling_objects_dataset.py` now validates `wan_num_frames + wan_prefix_frames` (new option,
  default 0 -> byte-identical behaviour for every existing config), because the model prepends the
  prefix before encoding, so the TOTAL is what must satisfy the rule.

**CPU verification passed before any GPU time:**
- `WanBlurCondEmbed` output is exactly 0.0 for random input -> enabling this pathway cannot perturb
  the resumed checkpoint at step 0.
- token count 800 == Wan's post-patch `t*h/2*w/2` for a 320x320, t_lat=8 latent.
- per-latent segment assignment as tabulated above, no mixing.
- val dataset builds at wan_num_frames=24 / wan_prefix_frames=5; n=3, N=24, r=8 exact.

**Not yet run**: the end-to-end GPU smoke test, because both cards are occupied (k038 = the user's
VAEBlurAlignFluxLoRA GoPro job, k040 = the data-only control arm of this very comparison). The control
is deliberately left running -- it is the baseline this arm is measured against.

**One caveat on the comparison**: this arm is NOT a pure ablation of the control. Enabling the
conditioning forces N=24/p=5 (so 3 segments partition evenly and the prefix is latent-aligned) where
the control is N=25/p=4. Token count and t_lat are identical (29 -> 8), but the dynamics-token
zero-padding covers 2 latents here vs 1 there, so a small part of any difference is layout, not
conditioning. A strict ablation would need a third run at N=24/p=5 with `burst_blur_conditioning:
false`; worth it only if this arm shows an effect.

## Fine-tune arm 1 (data only) COMPLETE; cross-attention arm implemented, pooling removed (2026-09-22)

**Arm 1 result -- no forgetting, weak learning.** The mixed Adobe240+GoPro+FMO/TbD fine-tune ran its
full budget (140000->150000, 4h45m, clean exit, loss 0.172 -> 0.0632, LR held at 2e-5 by the
after_resume_checkpoint override). checkpoint-149500 backed up to
`/depot/chan129/users/harshana/backups/Wan22VJEPADynamics_burst3_finetune_fmo_mixed_2026_09_21_17_19_05/`.
Control-flanked test on the in-loop validation videos at every 1k steps (140k = the resume point, so
a built-in before/after):

```
probe                       140k     150k    neg_ctrl  pos_ctrl   verdict
HELDOUT v_key              0.029    0.443     0.132     0.579     REAL improvement (sustained trend)
HELDOUT outf1              0.196    0.374     0.162     0.765     noise (preceded by 0.087, 0.157)
HELDOUT v_marker           0.114    0.124     0.120     0.682     flat at floor
HELDOUT out2               0.151    0.163     0.143     0.587     flat at floor
PRESERVE adobe 720p        0.580    0.632     0.270     0.751     retained
PRESERVE adobe GOPR9635    0.426    0.460     0.467     0.762     at floor, but was at baseline too
PRESERVE gopro 0384_11_00  0.597    0.562     0.081     0.687     retained
PRESERVE gopro 0384_11_05  0.754    0.635     0.220     0.809     retained (noisy)
```

**No catastrophic forgetting** -- every preserve probe stays above its negative control, so mixing
worked and adding GoPro did not cost Adobe240. **Learning: 1 of 4 held-out scenes.** Only v_key is
convincing (below-chance -> clearly above, sustained over the last five checkpoints, on a scene with
verified zero leakage). outf1's final 0.374 is preceded by 0.087/0.157 -- quoting it would be the
cite-the-best-number trap. A paired 28-sample eval (7 bursts per held-out scene, same samples for
baseline and fine-tune) is running to settle v_key/outf1 with real sample size.

**Pooling removed from the cross-attention path (user directive: "nothing should be pooled").**
Both pre-existing connectors destroy information before Wan sees it:
- `pooled`: `grid.mean(dim=(2,3))` annihilates the 24x24 spatial grid -> one vector per frame
  broadcast uniformly, i.e. no "where" at all.
- **both** `pooled` and `factored`: then temporally interpolate vjepa_num_tubelets ->
  post_patch_num_frames (15 -> 8 for burst-3), blending adjacent tubelets and smearing the very
  segment boundaries the burst construction exists to create.

The temporal resample is unavoidable for the value-residual path (its token count must equal Wan's),
but is entirely unnecessary for cross-attention, because the KV is a SET rather than a per-position
signal. So `dynamics_injection: cross_attention` now attends to the **raw V-JEPA grid, unpooled and
unresampled**: M = vjepa_num_tubelets * grid_h * grid_w = 15*24*24 = **8640** tokens for burst-3, all
three segments' tubelets kept separate and in order. The connector is applied pointwise so its trained
weights still contribute without collapsing anything.

Implementation notes worth keeping:
- Sizing matters here. A full-width cross-attention costs ~50M params/block = **1.5B** over 30 blocks,
  against the 11.8M LoRA beside it. Rebuilt with a low-rank bottleneck (dim->rank->dim, rank 64):
  **0.84M/block, 25.1M total**.
- Positional embedding is **factorized** (per-tubelet + per-spatial-cell, summed) rather than one flat
  8640-entry table, which would alone have cost ~16.6M over 30 blocks. Costs ~1.1M and encodes the
  structure explicitly. Without ANY positional signal the KV is an unordered bag and attention cannot
  distinguish tubelet 1 from tubelet 5 -- which would discard the burst ordering entirely.
- `to_out` zero-init verified exactly 0.0, so resuming a checkpoint into cross-attention mode starts
  unperturbed.
- Modules registered in `self.models` so they are actually checkpointed (the
  [[module_swap_checkpoint_compat]] trap: trainable-but-unregistered weights train and vanish).

Ops note: `nohup bash -c '...'` over ssh does NOT inherit the conda env -- the first paired-eval
launch died instantly on `ModuleNotFoundError: comet_ml`. Activate inside the subshell.

## Paired 28-sample eval OVERTURNS the arm-1 result: data-only fine-tune did NOT help (2026-09-22)

Ran baseline checkpoint-140000 and fine-tuned checkpoint-149500 over the **same 28 held-out samples**
(7 bursts per held-out scene, motion_square crops centred on real motion), so the comparison is paired
rather than one-sample-per-scene:

```
scene                n     BASE 140k       FT 149500      neg     pos      delta
HighFPS_GT_out2      7   0.280+-0.122    0.232+-0.143    0.139   0.615    -0.048  (3/7 up)
HighFPS_GT_outf1     7   0.236+-0.118    0.185+-0.082    0.182   0.773    -0.051  (3/7 up)
v_key_GTgamma        7   0.052+-0.067    0.118+-0.157    0.090   0.439    +0.065  (4/7 up)
v_marker_GTgamma     7   0.143+-0.114    0.194+-0.144    0.196   0.560    +0.051  (5/7 up)
ALL                 28   0.178+-0.139    0.182+-0.141                     +0.004  (15/28 up)

paired delta = +0.004 +- 0.032 SE,  t = 0.14,  15/28 improved (chance = 14/28)
```

**Null result.** The fine-tune produced no measurable improvement on held-out localized motion. Two
scenes drift slightly up, two slightly down, all within their own per-sample spread.

**This overturns the previous entry's headline claim.** That entry called v_key "REAL improvement
(sustained trend)" on the strength of its in-loop validation going 0.029 -> 0.443. With 7 samples
instead of 1, v_key is 0.052 -> 0.118 with a std of 0.157 -- the 0.443 was a lucky single draw. The
apparent "sustained trend" over the last five checkpoints was reading a trend into single-sample
noise. Do not cite the 1-of-4 claim; this table supersedes it.

Also note the absolute levels: BASE 0.178 and FT 0.182 both sit essentially AT the negative-control
band (0.090-0.196), far below the positive controls (0.439-0.773). Neither checkpoint renders the
motion. v_key and v_marker are actually *below* their own negative controls in both conditions; only
out2 is clearly above its floor, and it moved down.

**What this does and does not say.** It does NOT say the training data was useless -- forgetting was
successfully avoided (see the preserve probes in the previous entry, which stand: those were measured
per-checkpoint across 11 points, not a single draw, and all stayed well above their floors). It says
the *data gap alone* does not explain the failure, which strengthens rather than weakens the
architectural diagnosis: the pooled value-residual path cannot express "object at position x at time
t" no matter how much localized-motion data it is shown.

**Methodological lesson, third time in this project:** single-sample-per-scene in-loop validation is
not sufficient evidence for a quality claim here. The earlier 50/50-vs-0/8 reversal confusion, the
v_pen "tracks the trajectory" misread, and now this all have the same shape -- a number moved, and it
was noise. Run the paired multi-sample eval before claiming an effect.

## Cross-attention arm: three silent no-ops found before any result was believed (2026-09-22)

Implementing `dynamics_injection: cross_attention` hit THREE independent bugs that each made the new
branch a no-op while training looked perfectly healthy — loss descending, checkpoints writing, no
warnings. Recording them because the shape recurs in this codebase and a loss curve cannot detect any
of them.

**1. PEFT froze 89% of the module.** The cross-attn stack was attached to `WanBlockWrapper` as a
submodule, putting it inside `transformer.blocks`; `get_peft_model` then set `requires_grad=False` on
every base-model parameter. Worse, its submodules are named `to_q/to_k/to_v/to_out` — exactly the
`lora_target_modules` patterns — so PEFT *also* substring-matched and injected LoRA adapters INTO the
new module. Audit showed `WanDynamicsCrossAttnStack trainable 3,010,560 / 28,106,880 = 10.7%`, and
PeftModel inflated from 11,796,480 to 14,807,040 by the injected adapters. PGA escapes this only
because it is referenced from a plain non-`nn.Module` attention processor, so it never enters the
module tree. Fix: hold the stack on the shared context (`ctx.xattn[ell]`), same escape route as PGA.

**2. The module never reached the optimizer.** `_trainable_parameters()` hardcoded
`(dynamics_connector, modulation, pga)`. Both modules added this session (`dynamics_xattn`,
`blur_cond_embed`) were appended to `self.models` — so they SAVE, and `printize_model_memory` reports
them as trainable since it checks `requires_grad` — but were never handed to AdamW. Both are
zero-init, so they would have stayed exactly zero forever. Note fix 1 made this strictly worse before
it was caught: with the stack off the module tree it was no longer swept up by
`self.transformer.parameters()` either, taking it from "crippled" to "completely dead".
**`burst_blur_conditioning` was in this state from the moment it was written** — that arm would have
measured "blur conditioning doesn't help" while doing literally nothing.

**3. `ModuleList` checkpoint-folder collision.** Save/load hooks key folders on `_get_name()` +
occurrence counter, walking `self.models` in REVERSE. PGA is a bare `nn.ModuleList` → `ModuleList_1`
in every existing checkpoint. A second bare `nn.ModuleList` (the new stack), appended last, is popped
FIRST and claims `ModuleList_1`, shifting PGA to a `ModuleList_2` that does not exist. And the
loader's failure path only PRINTS — so PGA's 140k trained steps would have been silently discarded.
Fixed by subclassing purely to rename (`WanDynamicsCrossAttnStack`). That exposed a further latent
defect: a module absent from the checkpoint made both load attempts fail and the handler's own
f-string referenced an unbound `c`, raising OUT of the except block and aborting the whole load loop
(observed: only 1 `Loading model` line vs the healthy run's 4). Guarded with an `isdir` skip.

**VERIFICATION THAT ACTUALLY PROVES IT.** `to_out` is zero-init, so it gates the entire branch — if the
optimizer never moves it, output is exactly 0 regardless of what the loss does. Checkpoint-500 of the
from-scratch run:
```
0.to_out.weight  absmax=5.5518e-03     to_out layers nonzero: 30/30
0.pos_t          absmax=6.9087e-02     (burst-ordering embedding, frozen in the broken version)
Total trainable  41,724,294            xattn 25,096,320 / 25,096,320 = 100%
```
**Use this check, not the loss, for any zero-init adapter.** All three bugs above produced healthy loss
curves.

## MixedDataset sampled a fixed PREFIX, and mixed val metrics collapsed (2026-09-22)

Two further bugs, found by the bug-hunting agent and fixed on request.

**Prefix sampling.** `(i, j % len(ds)) for j in range(n_i)` — when `n_i < len(ds)` only sub-indices
`0..n_i-1` were reachable, and since the plan is built ONCE in `__init__` and reused for the whole run
(not rebuilt per epoch) the rest of the pool was unreachable *forever*: 52.5% of Adobe240 at the
shipped 4000x0.50 vs a 4214 pool, and because sources interleave scenes round-robin, the unreachable
half was the late part of every scene. Validation was worse — `n_i=2` meant burst 0 of the first two
scenes, every time, so the GoPro "unseen scenes" probe was two segments of ONE video whose siblings
are in training. Fixed per use_shuffle: training draws from the full pool at access time (mix ratio
still exact, since slot counts are fixed); validation uses evenly-spread deterministic indices
(verified: Adobe240 now 0 and 216 of 433; GoPro now two distinct source videos).
**This confounds arm 1's forgetting comparison** — it trained on a fixed early-timeline subset, not
the data its baseline saw. The paired 28-sample null result is unaffected (it used
FallingObjectsDataset directly).

**Metric collapse.** All val samples logged `video_psnr`/`video_ssim` under one untagged name at the
same step, so a tracker keeping one value per step showed whichever source was logged last — merging
the retention probe and the new-capability probe, which sit at very different absolute levels. Now
tagged `video_psnr/<source>`, with duplicate class names disambiguated (this project mixes TWO
FallingObjectsDataset sources; untagged they would have silently merged again).

## Tedla et al. (blur2vid) baseline extended: Falling Objects + GoPro (2026-09-22)

Ran `models/Baselines/Blur2Vid/blur2vid_lib/inference.py` on the two datasets this line of work
actually cares about, beyond the earlier Adobe240-only comparison (see the 2026-09-05 entries above).

**Checkpoint caveat (applies to both runs below):** `inference.py::load_model` hardcodes
`hf_hub_download(..., filename="cogvideox-outsidephotos/checkpoint/model.safetensors")` regardless of
`--model_config_path` — so both runs use the authors' general in-the-wild model, not a
GoPro-finetuned (`cogvideox-gopro-test`) or falling-objects-specific checkpoint. Both are therefore
pure generalization tests for blur2vid, exactly analogous to what this project's own Stage A oracle
checkpoint was subjected to on the same data.

**Falling Objects.** Built inputs with the new `analysis/prep_blur2vid_fallingobjects.py`, reusing
`FallingObjectsDataset`'s own motion_square crop and the exact peak-motion centers from
`Wan22VJEPADynamics_stageA_burst3_ar_infer_fallingobjects_peak.yml` (box=301, cell=156, key=270,
marker=313, pen=341, rubber=304) — same framing our own model was given, window_size=11 for both the
blur average and the GT frames. Ran on gilbreth-k040 (an already-running interactive allocation,
alongside its own training job — headroom confirmed before launching). Scored with
`analysis/compute_video_metrics.py` against the 11-frame GT (resized 1280x720, matching blur2vid's
fixed output resolution):

```
Tedla et al. 2025 (blur2vid, outsidephotos ckpt)   PSNR=37.76  SSIM=0.9588  LPIPS=0.0127  EPE=1.637  FVD(i3d)=1510.09  n=6, 6/6 reversed
```

**Read this with the same caveat already established for this dataset** (see the "Square-crop
(no-padding) high-res rerun" and "Controlled test of the only question that matters" sections above,
2026-09-21): a falling object occupies a tiny fraction of the frame against an otherwise-static
background, so whole-frame PSNR/SSIM/EPE mostly measure background reconstruction, not whether the
object's fall is actually rendered. Our own model's raw metrics on the identical scenes/centers/crop
(motion_square, peak-motion, checkpoint-140000) were PSNR=40.22 SSIM=0.9650 LPIPS=0.0106 EPE=0.640,
3/6 reversed — similar magnitude to blur2vid's numbers here — but the controlled moving-content
correlation test (GT vs GT+2f positive control, GT vs time-reversed-GT negative control) showed that
model does NOT actually render the object (scores at/below the negative-control floor in 4/6 scenes).
**Did not yet run that same controlled test on blur2vid's predictions** — the raw metrics above alone
are not evidence either way about whether blur2vid renders the object's motion; that correlation test
is the natural follow-up before drawing a comparative conclusion.

**GoPro.** Ran on the same 8 validation blur images (`GOPR0384_11_00/blur/000001..000008.png`) used
by `Wan22VJEPADynamics_bestckpt_infer_GoPro.yml`, for consistency with this project's own GoPro
qualitative check (see the 2026-09-07 "Qualitative-only check on MIOR and GoPro" entry). Qualitative
only — GoPro's local test split has no high-fps burst, so there is no real motion ground truth to
score a generated video against, same reason our own model's GoPro run was qualitative-only. All 8
videos generated successfully (~120KB each, visibly non-trivial content per file size).

Results copied to `/depot/chan129/users/harshana/Blur2Vid/tedla2025_blur2vid_fallingobjects/`
(blur_inputs/, gt/, predictions/, metrics.txt, metrics.json) and
`/depot/chan129/users/harshana/Blur2Vid/tedla2025_blur2vid_gopro_qualitative/` (blur_inputs/,
predictions/, README.txt).

## RESULT: cross-attention injection beats both baselines, paired 28-sample eval (2026-09-23)

Three models scored on the **identical 28 held-out samples** (7 bursts x 4 held-out scenes,
motion_square crops), same instrument that returned the null for the data-only arm:

```
scene              n   BASE140k  FT149500    XATTN     neg     pos
HighFPS_GT_out2    7     0.280     0.232     0.395    0.139   0.615
HighFPS_GT_outf1   7     0.236     0.185     0.353    0.182   0.773
v_key_GTgamma      7     0.052     0.118     0.187    0.090   0.439
v_marker_GTgamma   7     0.143     0.194     0.236    0.196   0.560
ALL               28     0.178     0.182     0.293

FT149500 vs BASE   delta=+0.004 +- 0.032 SE   t=0.14   15/28 up    (null, as before)
XATTN    vs BASE   delta=+0.115 +- 0.034 SE   t=3.42   21/28 up    SIGNIFICANT
XATTN    vs FT     delta=+0.111 +- 0.036 SE   t=3.08   20/28 up    SIGNIFICANT
```

**The data-only null reproduces exactly** (+0.004, t=0.14, 15/28 -- identical to the earlier run),
which validates the pipeline and instrument rather than just asserting them.

**Cross-attention is a real effect.** t=3.42 vs the pretrained baseline and t=3.08 vs the data-only
fine-tune, at n=28. This is the first positive result in the investigation that survives the paired
multi-sample test -- the same test that overturned the earlier "v_key improved" claim from
single-sample-per-checkpoint data.

**Interpretation, with the honest limits:**
- It confirms the architectural diagnosis. Adding localized-motion DATA to the pooled value-residual
  path did nothing (t=0.14); changing HOW the dynamics reach Wan moved it significantly. The
  bottleneck was the injection mechanism -- `grid.mean(dim=(2,3))` collapsing the 24x24 grid to one
  vector per frame, plus the 15->8 temporal interpolation -- not a shortage of training data.
- **Not a like-for-like ablation.** XATTN was trained FROM SCRATCH for 20k steps with cross-attention;
  FT149500 was a 10k fine-tune from the 140k checkpoint with value_residual. Architecture is the
  intended variable but the training regimes differ too, so some of the gap could be regime. A strict
  test would train value_residual from scratch for 20k on the same mix.
- **Far from solved.** 0.293 against positive controls of 0.439-0.773 means roughly 40-67% of the way
  to an "almost correct" rendering. v_marker (0.236 vs its 0.196 floor) is only marginally above
  chance; out2 and outf1 are the clear wins.
- Held-out scenes were chosen for measurability BEFORE training and never trained on; leakage
  verified zero.

Checkpoint-19500 loaded all five modules cleanly (including WanDynamicsCrossAttnStack_1), no shape
mismatches, no errors.

## 2026-09-23 -- Input resolution vs Wan2.2-TI2V-5B's native distribution

Triggered by the question "shouldn't Wan's input be 720p/1280x720, otherwise it
won't match the trained distribution?". The answer is **yes in part, but the
naive fix is wrong** -- and the reason is worth writing down because it is
counter-intuitive.

### What the checkpoint actually is

`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, from its own shipped configs:

| field | value | implication |
|---|---|---|
| VAE `scale_factor_spatial` | 16 | latent = px/16 |
| transformer `patch_size` | (1,2,2) | token = latent/2 |
| **=> one Wan token** | **32x32 px** | **at EVERY resolution** |
| `rope_max_seq_len` | 1024 | per-axis; no resolution is out of RoPE range |
| scheduler `flow_shift` | **5.0** | this is Wan2.2's *720p* setting (480p uses 3.0) |

TI2V-5B is the 720p-only member of the Wan2.2 family (the A14B pair do 480p and
720p); its official inference size is 1280x704. Note `1280x720` is NOT usable --
720/16 = 45 latent rows, which is odd and cannot be halved by patch_size 2.
**H and W must be divisible by 32.** 704 is the correct height, not 720.

### The key invariance: raising resolution does not add information

Because a token is always 32x32 px, feeding the *same crop* at a higher
resolution buys **more tokens on the object, but zero extra detail**. It is
worth doing only if the extra tokens are the bottleneck -- it is NOT a way to
recover pixels that the source never had.

So the discriminating question is not "what resolution does Wan want", it is
**"are we currently downscaling our sources?"**. Measured native sizes:

| source | native | fed to Wan | pixel scale |
|---|---|---|---|
| Adobe240 | **640x352** | random 320x320 crop | **1.00x (native)** |
| GoPro (GOPRO_Large_all) | 1280x720 | random 320x320 crop | **1.00x (native)** |
| FMO falling_objects | **300x960 (portrait)** | side=300 square crop -> resize 320 | 1.07x |
| TbD tbd3d | **960x400** | side=400 square crop -> resize 320 | 0.80x |

**We are not downscaling.** Every source reaches Wan at 0.80x-1.07x native
pixel scale. Three of the four sources *cannot* supply 720p pixels at all --
Adobe240 is a 640x352 dataset, FMO is 300px wide, TbD is 400px tall. Only GoPro
is natively 1280x720, and there we crop rather than downscale.

Therefore "train at 1280x704 to match the distribution" would, for 3 of 4
sources, mean **upsampling** -- paying 8.8x compute to interpolate pixels that
do not exist.

### What IS genuinely off-distribution

Not the pixel scale -- the **sequence length and token budget**:

- 320x320 -> latent 20x20 -> **10x10 = 100 spatial tokens**
- 1280x704 -> latent 80x44 -> **40x22 = 880 spatial tokens**

Wan's priors (how much a token "means", how attention spreads) were learned at
880. We run it at 11% of that. Per FMO's `gt_bbox`, objects are **20x37 to
108x92 px** native, i.e. **0.67 to 3.6 tokens** wide at 32 px/token -- so the
smallest (v_cell, v_rubber) are around or below one token, but the "~10px,
sub-token" framing in an earlier draft of this entry was overstated and is
corrected here. The token-budget argument still holds for the small end, and
upsampling DOES buy token coverage even though it adds no information.

`flow_shift: 5.0` is a second, independent symptom: that shift is calibrated
for 880 tokens. At 100 tokens it over-weights the high-noise end. Training here
uses **uniform** `t = torch.rand(bsz)` with no shift at all
(`Wan22VJEPADynamics_model.py:1050`) while sampling uses the scheduler's 5.0 --
a train/inference schedule mismatch that is free to fix and is *more* wrong the
further we are from 720p.

### Cost table

CORRECTED: the real frame count is wan_num_frames(25) + deblur_prefix_frames(4)
= 29, so latent T = (29-1)/4 + 1 = **8**, not 7. Absolute token counts below are
1/7 higher than first written; the RATIOS are unchanged and were always right.
Verified against both live configs: 320 -> 8x10x10 = 800 tokens, 512 -> 8x16x16
= 2048 tokens (2.56x). The V-JEPA KV side is 4 tubelets/segment x 3 segments x
24x24 = 6912 tokens at BOTH resolutions -- `vjepa_num_tubelets: 4` in the config
is the PER-SEGMENT count, which the model multiplies by burst_size internally.

| HxW | tokens h x w | total tokens | rel cost | measured/est s/it (solo) |
|---|---|---|---|---|
| 320x320 (current) | 10x10 | 800 | 1.0x | **1.79** (3.2 shared w/ Stage B) |
| 640x352 | 20x11 | 1760 | 2.2x | ~4-5 |
| 704x384 | 22x12 | 2112 | 2.6x | ~5-6 |
| 704x704 | 22x22 | 3872 | 4.8x | ~9-11 |
| 1280x704 (native) | 40x22 | 7040 | 8.8x | ~18-25 -> 50k steps = 10-14 days |

Attention is quadratic in tokens, so wall-clock scales worse than the token
ratio. 1280x704 is not viable on 4h standby slots. Memory is NOT the binding
constraint -- Stage A at 320 uses only 9.7 GB reported / 28 GB resident with
41.7M trainable params.

**Recommendation: 640x352.** Exactly half of 1280x704, so it is 16:9 and
/32-clean; 2.2x cost; 2.2x the token budget; and it is Adobe240's *exact native
size*, so that source stops being cropped at all.

### V-JEPA geometry under a 16:9 Wan input

`pos_s` is hardcoded (576,64) = 24x24, so V-JEPA stays 384x384. Do **not**
letterbox-pad a 16:9 crop into that square -- ~44% of the grid would be black
bars, discarding over a third of the dynamics tokens the cross-attention exists
to carry. Use an anisotropic resize; `pos_s` and the connector are learned and
absorb the aspect distortion.

### CORRECTION: 640x352 is the WRONG target -- FMO's motion is portrait-shaped

Measured from `falling_objects/gt_bbox/*.txt` (col0 = x in [10,196], col1 = y in
[268,852], cols 2-3 = object w,h), over every 27-frame burst window:

| scene | n | obj w x h | max x-travel / burst | **max y-travel / burst** |
|---|---|---|---|---|
| v_box | 176 | 76x41 | 16 | **229** |
| v_cell | 152 | 20x37 | 18 | **249** |
| v_key | 208 | 40x43 | 96 | **235** |
| v_marker | 88 | 46x86 | 9 | **248** |
| v_pen | 144 | 108x92 | 50 | **290** |
| v_rubber | 184 | 46x46 | 34 | **222** |

A burst needs **>=290 px of vertical window plus up to 92 px of object height
~= 380 px tall, inside a frame only 300 px wide** -- a portrait motion envelope
(aspect ~0.79). **No 640x352 canvas can hold it at any pixel scale**: you either
crop away most of the fall or squash it ~2.3x. Today's 300x300 square crop works
only because it *barely* covers the 222-290 px burst displacement.

The "Wan must be 16:9" premise is also weaker than it looked -- Wan's RoPE is
factorized per axis with `rope_max_seq_len: 1024`, and Wan2.2 ships portrait
704x1280 as a supported size. **Token budget is the real constraint, not aspect.**

**Revised recommendation: 512x512.** 32x32 latent -> 16x16 = 256 spatial
tokens/frame -> 8x16x16 = 2048 total = 2.56x current (VERIFIED against the config), the same cost band as 640x352 (2.2x), ~4-6 s/it.
- FMO 300x300 native crop -> 512 (1.71x upsample); trajectory coverage unchanged
- Adobe240 352 -> 512 (1.45x), whole frame, zero crop loss
- GoPro native-scale 512^2 crop (larger window than today, pixel scale kept)
- TbD 400 -> 512 (1.28x)
- V-JEPA stays 1:1, so `pos_s` (576,64) needs no aspect reinterpretation and the
  letterbox-vs-squash question disappears

## 2026-09-23 -- RETRACTED: "58% of FMO training samples contain no object"

**This claim was wrong. Do not cite the 58% / 15.1% / 28.6% figures.**

They came from indexing `gt_bbox/<scene>.txt` row *i* as frame *i* of
`imgs_gt/<scene>/`. That alignment is invalid: `imgs_gt` holds **500** frames
per scene (00000000..00000499) while `gt_bbox` holds only **88-208** rows,
varying per scene. The annotation timeline is offset and/or subsampled by an
unknown factor, so every bbox-derived number was meaningless -- including the
follow-up claim that `_motion_square_offset` returns a near-constant top of
~392 and is therefore broken.

**Lesson (same family as [[dictasmember_silent_empty_dict]]): a side file that
loads and indexes without error is not a side file that is aligned.** Check row
counts against frame counts before trusting any annotation join.

### The valid, annotation-free re-measurement

Detect the moving object directly: `dev = |frame - temporal_median(span)|` over
each 27-frame burst, threshold at `max(0.06, 0.35 * p99.9(dev))`, and ask
whether the crop window contains moving content. 24 FMO bursts, 6 scenes:

| crop mode | permissive (any moving row inside) | strict (all moving rows inside) |
|---|---|---|
| `random_square` (shipped) | **87.8%** | 3.5% |
| `motion_square` | 100% | 9.4% |
| `motion_jitter +-24` (new) | -- | 9.1% |
| best possible placement | 100% | **10.6%** |

**`random_square` is not broken.** The strict column's ceiling is only 10.6%,
which means that metric is dominated by isolated above-threshold rows (a single
noisy row anywhere breaks containment), so it cannot carry a decision either.

A `motion_jitter` crop mode (motion-energy offset + configurable jitter) WAS
implemented in `FallingObjectsDataset` and is available, but is **not enabled**
in any config: with no clear evidence it helps, enabling it would confound the
512 resolution change with a second simultaneous variable.

## 2026-09-23 -- Stage B (single_clip) finished, but it CANNOT feed Stage C yet

`BlurJepaViT_stageB_singleclip_mixed_2026_09_22_21_47_40` completed
20000/20000 (12h31m, 2.25 s/it); latest surviving checkpoint is **19500**
(`checkpoints_total_limit` rotated the rest). The process exited cleanly.

It is **not** chainable into the new Stage A. Three mismatches, one of them fatal:

| | Stage B (trained) | Stage A (512 run) | severity |
|---|---|---|---|
| `vjepa_window_size` | **10** -> 5 tubelets/segment, **15 total** | **8** -> 4/segment, **12 total** | **FATAL** |
| input resolution | 320 | 512 | soft |
| FMO train crop | `random_square` (58% empty) | `motion_jitter` | quality |

The tubelet count is a hard shape contract: a 15-tubelet predictor output into a
12-tubelet cross-attention stack raises the same `ValueError: got N=6912 tokens,
expected 8640` seen on 2026-09-22. **Stage C cannot run until Stage B is
retrained at `vjepa_window_size: 8`.**

The resolution mismatch is softer because V-JEPA always resizes its input to
384x384, but the *object's pixel scale inside that 384 crop* differs: a 320 crop
is upsampled 1.2x, a 512 crop downsampled 0.75x, so the predictor would see
objects ~1.6x smaller than it was trained on. Worth matching.

Stage B's cost is dominated by V-JEPA at 384x384 and is essentially independent
of the Wan-side resolution, so a 512 + win8 + motion_jitter retrain should again
be ~12h for 20k steps.

### Launch record (2026-09-23, both on gilbreth-k040)

| | Stage A 512 | Stage B win8+512 |
|---|---|---|
| config | `Wan22VJEPADynamics_stageA_xattn_512.yml` | `BlurJepaViT_stageB_singleclip_win8_512.yml` |
| start | warm from `..._win8_2026_09_23_01_27_19/checkpoint-29000` | from scratch |
| steps | 29000 -> 50000 (21000 new; max_train_steps is ABSOLUTE) | 0 -> 20000 |
| s/it (shared GPU) | 5.83 | 2.05 |
| GPU | 31.3 GB | 18.8 GB (50.1/80 total) |
| comet | 3cc7ab8f23c64451a50729f9976318a7 | -- |

Stage A resume verified: all 5 modules (`WanDynamicsCrossAttnStack_1`,
`ModuleList_1`, `TimestepLayerModulation_1`, `JEPAPerceptualEncoder_1`,
`PeftModel_1`) loaded with **no SHAPE MISMATCH** -- unlike the 15->12 tubelet
transition, a resolution change touches no trainable tensor (the xattn's
pos_t/pos_s are on the V-JEPA side, `to_q` is `Linear(dim, rank)`). Trainable
params 41,718,534, unchanged. First-step loss 0.223, within the 320 run's
0.115-0.306 band -- the predicted warm-start spike did not materialise.

Stage B confirms win8: `V-JEPA output shape verified: [B, 2304, 1024]
(4 tubelets x 24x24 spatial)` -> 4x3 = 12 total, matching Stage A.

NOTE on `use_same_dir`: it controls only the OUTPUT directory
(`base_model.py:189-197`), NOT whether a checkpoint loads. Warm-starting from
another run's directory into a fresh timestamped output dir works without it,
as both these runs demonstrate. The memory note about `resume_from_path` being
silently ignored applies to continuing a run IN PLACE, not to warm starts.

## 2026-09-23 -- Dynamic burst size (S sampled per step, 1..5)

Motivation (user): when the number of blur frames changes, the V-JEPA input
changes too -- and since cross-attention's KV is a **set**, its length never has
to match Wan's token count, so the xattn should be able to absorb a varying
burst size. That is correct. Exactly ONE parameter pinned it:
`WanDynamicsCrossAttn.pos_t`, shape `(n_tubelets, rank)`, with `_kv_pos`
**raising** unless `m == n_tubelets*grid_h*grid_w`.

### Design: datasets emit the MAX burst, the model truncates per step

Verified empirically, not assumed: segment *i* occupies `burst_frames[i]` and
full-span frames `[stride*i, stride*i + window_size)`, and **segment i's first
frame sits at `wan_frames` offset `stride*i - 1` with ZERO residual**. So the
first S segments of a max-burst sample are themselves a valid burst-S sample,
and the matching GT video is the first `F_S` frames:

```
F_S = wan_num_frames_max - center_stride * (S_max - S)
```

| S | F_S | (F+4-1)%4 | t_lat | Wan tokens | tubelets | V-JEPA KV |
|---|---|---|---|---|---|---|
| 1 | 9  | 0 | 4  | 1024 | 4  | 2304 |
| 2 | 17 | 0 | 6  | 1536 | 8  | 4608 |
| 3 | 25 | 0 | 8  | 2048 | 12 | 6912 |
| 4 | 33 | 0 | 10 | 2560 | 16 | 9216 |
| 5 | 41 | 0 | 12 | 3072 | 20 | 11520 |

A **startup guard** rejects any (wan_num_frames, center_stride, S-range) combo
where some reachable S violates Wan's `(F + prefix - 1) % 4 == 0`, turning a
rare mid-run crash into a launch-time error.

CAUTION: `burst_blur_conditioning` (currently DISABLED) requires
`wan_num_frames % burst_size == 0`. F_S = 8S+1 is never divisible by S for
S>1, so enabling it together with dynamic_burst needs that path reworked first.

### Changes

- `wan_pga.py`: `pos_t` allocated at `max_tubelets` and **sliced** per forward;
  `_kv_pos` now DERIVES the tubelet count from the KV length, enforcing only
  divisibility by the spatial grid and the max bound.
- `wan_wrappers.py` / model: `max_tubelets` threaded through the installer,
  sized `per_segment_tubelets * dynamic_burst.max`.
- `Wan22VJEPADynamics_model.feed_data`: the single choke point -- samples S,
  truncates `burst_blur`/`burst_frames`/`wan_frames`, sets `current_burst_size`.
- The two `range(self.burst_size)` loops in `_dynamics_full_grid_burst` now read
  `s_count` from the tensor (`self.burst_size` is only the MAXIMUM now).
- `BlurJepaDynamics_model`: same sampler + truncation. Everything downstream was
  already S-agnostic (`s = burst_blur.shape[1]`), so no other change.

### Schedule

Upper bound ramps `min -> max` linearly over `ramp_steps`; within the allowed
range `P(S) ~ S^-weight_power`. Measured with ramp 10000, power 1.0:

```
step      0: S=1:1.00
step   2500: S=1:0.68  S=2:0.32
step   5000: S=1:0.52  S=2:0.34  S=3:0.15
step  10000: S=1:0.41  S=2:0.23  S=3:0.15  S=4:0.11  S=5:0.10
validation: pinned at S=5 (mixing sizes would make the val curve incomparable)
```

S is drawn from `default_rng(seed + global_step)`, so **every DDP rank picks the
same S without a collective** -- ranks disagreeing would produce mismatched
shapes and deadlock the all-reduce.

### Validated (CPU, in isolation -- NOT yet a full GPU forward)

- `WanDynamicsCrossAttn` at 4/8/12/16/20 tubelets: correct output shape every
  time; over-max and non-grid-divisible KV both correctly rejected.
- Sampling: ramp as tabulated, deterministic per step, val pinned.
- Truncation: exact F_S at every S, input dict left unmutated.

**Still unverified: a real forward through the full model at varying S**, which
needs a free GPU. Not run because the card is occupied -- see
[[measure_gpu_footprint_before_colocating]]. Note S=5 is 1.5x the Wan tokens and
1.67x the KV of S=3, so peak memory will EXCEED the current 31.3 GB run.

### Checkpoint compatibility

Warm-starting the dynburst config from the fixed-burst-3 512 run is safe except
`pos_t` (12,64) -> (20,64), which base_model's shape filter SKIPS (one
"SHAPE MISMATCH, not loaded: N.pos_t" per block -- expected). Everything else
loads. pos_t is ~0.1% of the module and its temporal semantics change under a
varying burst anyway.

Configs: `Wan22VJEPADynamics_stageA_xattn_512_dynburst.yml`,
`BlurJepaViT_stageB_singleclip_win8_512_dynburst.yml` (schedules MUST match).

## 2026-09-23 -- Audit: pooling / interpolation / broadcasting on V-JEPA tokens

Full trace from token production to consumption.

**Source (`jepa_helpers.extract_vjepa_tokens`)**: `model(pixels).last_hidden_state`
then a pure `rearrange("b t h w d -> b (t h w) d")`. **No pooling, no CLS
selection, no mean** -- the full patch grid reaches the caller.

**Stage B (`BlurJepaDynamics_model`)**: **nothing** is pooled, interpolated or
broadcast on the tokens. Prediction and target are compared at full
`[B, 4S*576, 1024]`. The greps that look like reductions are not:
`.repeat(1,1,r,1,1)` acts on INPUT PIXELS (the still->clip trick, pre-V-JEPA),
`grid.flip(dims=[1])` is a reversible permutation of the tubelet axis, and the
`.mean(dim=(1,2))` calls are the MSE loss reduction. `tokens.mean(dim=0)` and
`F.interpolate(scale_factor=cell)` are PCA-panel display only.

**Stage A / C, ACTIVE path (`dynamics_injection: cross_attention`)**: **nothing.**
`_build_dynamics_kv` is `self.dynamics_connector(grid)` -- a pointwise
`nn.Linear(1024->3072)`. `to_k`/`to_v` in `WanDynamicsCrossAttn` are pointwise
too. All `4S*576` tokens stay individually addressable; the only expansion is
`_kv_pos` broadcasting the POSITION tables (`pos_t[:,None,:] + pos_s[None,:,:]`),
which builds the positional encoding, not duplicated content.

**Stage A / C, INACTIVE path (`value_residual`)** -- where all of it lives:

| function | operation | effect |
|---|---|---|
| `pool_dynamics_to_frame_tokens` | `grid.mean(dim=(2,3))` | spatial **576 -> 1**, total collapse |
| same | `F.interpolate(mode="linear")` | temporal `4S -> post_patch_frames`, blends across segment boundaries |
| `resample_dynamics_grid` | `F.interpolate(mode="bilinear")` | spatial 24x24 -> 16x16 (576 -> 256) |
| `broadcast_frame_tokens_to_video_tokens` | `.expand(-1,-1,256,-1)` | one vector copied to every spatial position |

Skipped under cross-attention since the 2026-09-23 fix; `dynamic_burst` +
`value_residual` now raises at startup.

### The one spatial reduction still live: 512 -> 384 before V-JEPA

`vjepa_from_unit_tensor` resizes the square crop **512x512 -> 384x384** (bicubic,
antialias) before V-JEPA. Irreversible, and upstream of tokenization. Note what
this implies: **raising the Wan canvas 320 -> 512 gave Wan 2.56x more tokens but
gave V-JEPA nothing** -- the dynamics branch has been at 384 throughout.

Granularity, which is the opposite of the intuitive guess:
- V-JEPA: 384/24 = **16 px per spatial token**, 576 positions
- Wan @512: 512/16 = **32 px per spatial token**, 256 positions

So V-JEPA's grid is 2x FINER than Wan's. Cross-attention needs no resampling to
reconcile them precisely because its KV is a set.

**DECISION (user, 2026-09-23): keep `vjepa_image_size: 384`. Do not raise it.**
V-JEPA 2.1 ViT-L/384 is trained at 384 and performs best there; going higher is
off-distribution and would require interpolating its positional embeddings for
no guaranteed gain. If small-object detail in the dynamics branch ever needs
improving, the lever is a TIGHTER CROP (more of the 384 frame occupied by the
object), not a larger V-JEPA input. Do not revisit this without new evidence.

## 2026-09-23 -- 512 vs 320, held-out paired eval: INCONCLUSIVE (do not cite as validation)

First actual test of the resolution change. Arms: the 512 run @checkpoint-31000
vs the 320 run @checkpoint-29000, same 4 held-out scenes, same
`centers_around_peak: 32`, 28 paired clips. Scored with
`analysis/score_control_flanked.py` (moving-content correlation against a
time-reversed negative control and a +2-frame positive control, normalized per
clip so two passes at different resolutions are comparable).

| statistic | 320 @29k | 512 @31k | delta | significance |
|---|---|---|---|---|
| normalized mean | 0.446 | 0.656 | **+0.210 +- 0.113** | t=1.86 |
| raw correlation | 0.368 | 0.441 | +0.073 +- 0.044 | t=1.66 |
| median normalized delta | | | +0.098 | (mean +0.210 -> outlier-inflated) |
| sign test | | | 17/28 up | p=0.172 |
| Wilcoxon normalized / raw | | | | p=0.144 / p=0.157 |

**Verdict: directionally favourable, NOT significant on any test.** All five
statistics point the same way, which is mildly encouraging, but every p-value
sits at 0.14-0.17. **512 is neither validated nor refuted.** For contrast, the
cross-attention result that this project's direction rests on was +0.115 at
**t=3.42** with 21/28 up -- that is what significance looks like here.

Three reasons not to read more into it:
1. **Confounded.** 31000 vs 29000 on the same lineage (the 512 run warm-started
   FROM the 320 run's checkpoint-29000), so this is "2000 steps at 512 vs the
   state it branched from", not a clean resolution ablation.
2. **Barely adapted.** The 512 model had only 2000 steps at the new resolution.
3. **High clip variance.** Per-clip normalized scores span -0.797 to +2.142.
   The control gaps are healthy (median 0.450), so this is genuine
   scene-to-scene variance, not a denominator artifact -- n=28 is simply not
   enough to resolve a delta this size.

**What would settle it:** re-run the same paired eval once the dynburst run has
a well-trained 512 checkpoint, against the 320 @29000 arm which is already
scored and on disk. Keep
`Wan22VJEPADynamics_stageA_xattn_win8_2026_09_23_01_27_19/checkpoint-29000`
until then.

Outputs: `experiments/infer/HeldoutMulti_{512_ckpt31000,320_ckpt29000}_*` (196
videos each). Ops note: each arm peaked at **63 GiB of the 64 GiB Slurm cgroup**
while sharing k038 with Stage B -- do not run a third job alongside.

## 2026-09-23 -- STAGE C VALIDATED: predictor == oracle, and dynamics does ALL the motion work

Three arms, all at the SAME Stage A checkpoint (dynburst @37000), same 28
held-out clips, differing only in where the dynamics tokens come from. Scored
with `analysis/score_control_flanked.py`.

| arm | raw corr | normalized |
|---|---|---|
| **no dynamics** (`disable_dynamics_injection: true`) | **0.127** | **~ -0.08** |
| **oracle** (`vjepa_gt`, real frames) | 0.326 | 0.416 |
| **Stage C** (`blur_jepa`, Stage B @20000 from blur only) | **0.335** | 0.411 |

| comparison | delta (raw) | t | up | p |
|---|---|---|---|---|
| oracle vs no-dyn | +0.198 +- 0.030 | **6.58** | 25/28 | <0.001 |
| Stage C vs no-dyn | +0.208 +- 0.030 | **6.90** | 27/28 | <0.001 |
| oracle vs Stage C | -0.005 +- 0.063 | -0.08 | 17/28 | 0.31 |

### Two conclusions, both strong

1. **The dynamics branch does essentially ALL the motion work.** With injection
   disabled the model lands at ~ -0.08 normalized -- AT OR BELOW the
   time-reversed negative control, i.e. no genuine motion, the exact signature
   of the 2026-09-21 OOD failure. Turning dynamics on moves it to ~0.41. This
   also retires the worry that the cross-attention numbers might be measuring
   the blur-conditioned I2V path rather than the dynamics.
2. **Stage B's predictor fully substitutes for the oracle.** Not "close" --
   statistically indistinguishable (t=-0.08) and nominally AHEAD on raw
   correlation (27/28 clips up vs no-dyn, against the oracle's 25/28). The
   deployable chain costs nothing relative to its upper bound.

### Why the control arm was necessary

Oracle == Stage C is ambiguous on its own: either the predictor is excellent, or
Wan barely uses the tokens. The two readings imply opposite next steps (ship it
vs. debug the injection). The no-dynamics arm discriminates, and it cost 10
minutes. **Never report an "A == B" result on a conditioning pathway without the
"neither" control.**

### Consequence for the roadmap

Stage B is NOT the bottleneck -- it is already at the oracle ceiling, so more
Stage B training buys nothing. **The ceiling is Stage A**: the oracle itself only
reaches 0.416 normalized (0 = reversed-motion floor, 1 = a 2-frame-shifted GT).
Effort should go to raising the oracle, not the predictor.

Arms on disk: `experiments/infer/Heldout_{ORACLE,STAGEC,NODYN}_dyn37000*`.

## 2026-09-23 -- RETRACTED: the burst-size sweep ran with ZERO trained weights

**The S=1/3/5 sweep numbers are void.** Root cause, from the logs:

```
### logs_sweep_S3.log
  modules loaded: 0
  Failed to load state  Tried to find .../checkpoint-37000 but folder does not exist
```

The sweep configs pointed at `checkpoint-37000` of the **still-training** Stage A
run. By the time the sweep started ~2h later, `checkpoints_total_limit: 3` had
rotated it away (the dir then held 0, 40000, 41000, 42000). `base_model.train()`
CAUGHT the exception, printed it, and continued -- so all three arms ran with
**initialisation-value adapters**. Because the adapters are zero-init by design
(cross-attention `to_out`, LoRA `B`), the model degrades to base Wan and still
emits plausible-looking video.

**The user caught this by eye before the statistics were believed** -- "visible
artifacts similar to the visual output close to step 0" -- which is literally
what it was. The quantitative signature was there too and I under-read it:

| run | GT energy | PRED energy | ratio | modules |
|---|---|---|---|---|
| training-run validation (live model) | 0.02507 | 0.01806 | **0.72** | n/a |
| eval Stage C 512 | 0.00242 | 0.00320 | **1.32** | 5 |
| eval NoDyn 512 | 0.00242 | 0.00372 | 1.54 | 5 |
| eval sweep S=3 | 0.00262 | 0.02486 | **9.49** | **0** |

A prediction/GT motion-energy ratio near 1 is healthy; **~9-11x means the run is
degenerate**, not merely worse. Add this ratio to any eval sanity check.

### Full audit of every eval run today

VALID (5 modules, 0 failures): `320_ckpt29000`, `512_ckpt31000`, `nodyn`,
`oracle_dyn37000`, `stagec_dyn37000`, `stagec_ckpt0`, `stagec_smoke`.
INVALID (0 modules): `sweep_S1`, `sweep_S3`, `sweep_S5`.

**So the headline results are unaffected**: dynamics does the motion work
(t=6.58), Stage C == oracle (t=-0.08), and 512-vs-320 inconclusive -- all rest
on runs that loaded correctly. Only the burst sweep is retracted.

### Fixes applied

1. **A failed checkpoint load is now FATAL** (`base_model.py`). Continuing was
   never recoverable: the model runs with init-value adapters and produces
   output that looks trained. It now raises with a message naming the rotation
   cause.
2. **Frozen eval checkpoints**: `experiments/Motion/EVAL_FROZEN_stageA_dynburst/`
   holds copies (399 MB each) outside any training run, immune to rotation.
   **RULE: never point an eval at a checkpoint inside a run that is still
   training.**
3. Note the round-trip check added earlier is gated on `self.is_train`, so eval
   runs skip it -- which is exactly where this failure hid. The fatal-raise
   closes that gap for the load path.

### Separately, the sweep was never viable at that checkpoint anyway

With `ramp_start_step: 31000` / `ramp_steps: 10000`, step 37000 gives
`hi = 1 + round(0.6*4) = 3`. **S=4 and S=5 had never been trained.** Check the
ramp state before designing a burst-size sweep.

## 2026-09-24 -- HIRES (736) three-way: main result CONFIRMED, one 512 claim CORRECTED

Re-ran oracle / Stage C / no-dynamics at **out_size 736**, the project's
REPORTING resolution for FMO/TbD (see [[check_results_means_hires]]). All three
from the FROZEN `EVAL_FROZEN_stageA_dynburst/checkpoint-58000`, 28 paired clips.
All arms verified: 5 modules loaded, 0 errors, motion-energy ratios 0.98-1.13.

| arm | raw corr @736 | raw corr @512 (ckpt-37000) |
|---|---|---|
| no dynamics | **0.210** | 0.127 |
| oracle | **0.426** | 0.326 |
| Stage C | **0.393** | 0.335 |

| comparison @736 | delta | t | up | p |
|---|---|---|---|---|
| no-dyn -> oracle | +0.216 +- 0.041 | 5.28 | 25/28 | <0.001 |
| no-dyn -> Stage C | +0.183 +- 0.042 | 4.39 | 24/28 | <0.001 |
| **oracle -> Stage C** | **-0.033 +- 0.011** | **-3.11** | **8/28** | **0.005** |

### Confirmed: the dynamics branch does the motion work

Holds at the reporting resolution and is stronger there in absolute terms
(0.426 vs 0.326 at 512), despite 736 being OFF the 512 training distribution.

### CORRECTION to the 2026-09-23 entry

That entry said Stage C is "statistically indistinguishable" from the oracle
(t=-0.08 at 512) and concluded "the predictor costs nothing measurable."
**At 736 that is false**: Stage C is significantly BELOW the oracle
(-0.033, t=-3.11, p=0.005, losing on 20 of 28 clips). The 512 eval simply lacked
the resolution to resolve the gap.

Revised statement: **Stage C recovers ~85% of the dynamics benefit**
(+0.183 of the oracle's +0.216) -- a good deployable result, but NOT parity.
Do not cite "predictor == oracle".

Consequence for the roadmap: the earlier claim that "Stage B is not the
bottleneck, do not train it further" was based on the 512 parity result and is
now weaker. There IS a real predictor gap to close, though it is small relative
to the oracle's own distance from the ceiling (0.426 of a normalized 1.0).
Stage A remains the larger lever, but Stage B is no longer provably maxed.

**Methodological lesson: a null result at the training resolution is not a null
result.** Verify "A == B" claims at the resolution you report at, before
concluding that a component is free.

## 2026-09-24 -- FINAL: Stage A dynburst @69000 (uniform-weighted), hires 736

Stage A dynburst finished (69999/70000). Weighting was flattened to uniform
(`weight_power: 0.0`) from step 58000. **Defect: the final ~1000 steps are
UNSAVED** -- `base_model` saves only when `global_step % checkpointing_steps == 0`
and the loop ended at 69999, so the last checkpoint is 69000, not 70000. Add a
save-on-exit before the next long run.

Frozen at `experiments/Motion/EVAL_FROZEN_stageA_dynburst/checkpoint-69000`.
Stage B = checkpoint-20000. 28 paired held-out clips, out_size 736, all arms
verified (5 modules loaded, energy ratios 0.94-1.27).

| arm | raw corr | @58000 |
|---|---|---|
| no dynamics | **0.156** | 0.210 |
| Stage C (predicted) | **0.439** | 0.393 |
| oracle | **0.476** | 0.426 |

| comparison | delta | t | p |
|---|---|---|---|
| no-dyn -> oracle | +0.319 +- 0.028 | **11.37** | <0.001 |
| no-dyn -> Stage C | +0.283 +- 0.034 | **8.44** | <0.001 |
| oracle -> Stage C | -0.036 +- 0.022 | -1.64 | 0.104 |
| oracle 58k -> 69k | +0.049 +- 0.044 | 1.13 | 0.39 |
| Stage C 58k -> 69k | +0.047 +- 0.044 | 1.06 | 0.21 |

**Confirmed, strongly:** the dynamics branch does the motion work (t=11.4 / 8.4).
The model also leans HARDER on it as training proceeds: the no-dynamics arm
FELL 0.210 -> 0.156 while the oracle rose 0.426 -> 0.476.

**Predictor gap -- do NOT read as "closed".** The 58k gap was -0.033 (t=-3.11,
p=0.005); the 69k gap is -0.036 (t=-1.64, p=0.10). The POINT ESTIMATE is
unchanged; the standard error doubled (0.011 -> 0.022). This is a noisier
measurement of the same gap, not evidence of convergence. Stage C recovers ~89%
of the dynamics benefit (0.283/0.319) vs ~85% at 58k -- same direction, within
noise. Claim "Stage C recovers ~85-89% of the oracle's dynamics benefit"; do not
claim parity.

**58k -> 69k gains are NOT significant** (oracle t=1.13, Stage C t=1.06). That
window bundles 11k more steps AND the switch to uniform burst weighting, so even
a significant gain could not be attributed to the weighting. The burst-size
sweep is the direct test and still needs re-running on this checkpoint.

Ops: the first Stage C attempt died with exit 137 (cgroup RAM SIGKILL, shard
3/5) because an unrelated `reFlux/StageB_GoPro.yml` trainer (32 GB, same user,
~1h old) was sharing k038's 64 GB cgroup. Re-run on idle k040 succeeded.

## 2026-09-24 -- FIXED: the final training step was never saved

**Bug:** `base_model.train()` saves only inside the loop, at the START of an
iteration on the PRE-increment `global_step` (so `checkpoint-N` = "N steps
completed"). The state reached by the LAST iteration is never checked against a
boundary, so a finished run's newest checkpoint could be up to
`checkpointing_steps` behind. The 58000->70000 Stage A resume ended by epoch
exhaustion (`first_epoch = 58000//2000 = 29`, `num_train_epochs = ceil(70000/2000)
= 35` -> exactly 12000 iterations, last check at global_step 69999) and the newest
checkpoint was 69000; ~1000 steps were lost. Unrecoverable -- 69000 is the final
Stage A model.

**Fix (`models/base_model.py`):** after the loop and BEFORE the final validation,
save `checkpoint-{global_step}` (skipped if not `is_train`, `global_step <= 1`, or
the directory already exists). Before the validation so the weights survive a
validation crash.

**Verified** with `checkpoints/MotionBlur2Video/BlurJepaViT_saveonexit_TEST.yml`
(4 steps/epoch, max 8, `checkpointing_steps: 1000` so the in-loop save cannot fire):
exit 0, `Saved FINAL state to .../checkpoint-8`, dirs = {checkpoint-0, checkpoint-8}.
Keep that config as a regression test. Not yet observed: that an EVAL run
(`is_train: false`) writes no checkpoint -- confirm on the next eval.

## 2026-09-25 -- Burst-size sweep on Stage A @69000 (Stage C path), 512: extra blur frames do NOT hurt; the S=5 full-length drop is a LENGTH effect

Valid rerun (frozen `checkpoint-69000`, 5 modules loaded on every arm, energy
ratios 0.98/1.15/1.16). Dataset emits burst-5; model truncates to S. n=19-20
paired clips (fewer than the 28 of the burst-3 evals -- a 5-segment burst needs a
longer span). RAW correlation is used throughout: at 9 frames the normalized
score has SE ~0.44 (control gap collapses), i.e. is unusable.

| S | frames | raw corr, FULL length | raw corr, COMMON 9-frame window |
|---|---|---|---|
| 1 | 9  | 0.410 | 0.410 |
| 3 | 25 | 0.446 | 0.448 |
| 5 | 41 | 0.288 | 0.428 |

| comparison | full length | common 9-frame window |
|---|---|---|
| S1 -> S3 | +0.036, t=0.65, p=0.71 | +0.037, t=0.77, p=0.35 |
| S3 -> S5 | **-0.152, t=-2.68, p=0.009** | -0.020, t=-0.37, p=0.35 |
| S1 -> S5 | **-0.115, t=-2.57, p=0.029** | +0.018, t=0.46, p=0.62 |

**Reading:** at full length S=5 is significantly worse, but scored on the SAME 9
GT frames it is NOT (-0.020, n.s.). The drop is therefore about generating a
41-frame video (or about the metric behaving differently on longer clips -- the
moving-content map uses a per-clip temporal median, so the two cannot be fully
separated here), not about a larger burst degrading the model. Nothing is
significant on the common window, and with SE ~0.05 this design detects
differences of roughly >=0.1 only -- so "extra blur frames neither help nor hurt
by >0.1", NOT "they are equivalent".

Superseded: the earlier (2026-09-23, ckpt-42000) sweep, where S=5 (1.8% of
training) scored 0.206 -- that was starvation, not burst size.

Tooling: `analysis/score_burst_sweep.sh <key-suffix> <label>` scores a sweep at
full length and on the common window. (The sweep chain script that first ran this
had inverted shell return logic and skipped scoring of valid arms.)

## 2026-09-25 -- Adobe240 + GoPro held-out evals (camera-shake), hires 1280x736, Stage A @69000

All 6 arms valid (5 modules loaded, 0 errors). 20 clips per dataset, spread evenly
over the test scenes (Adobe240 test: 8 scenes / 433 samples; GoPro test: 11 scenes /
501 samples), whole-frame resize to 1280x736. Oracle / Stage C (Stage B @20000) /
no-dynamics, all from frozen `checkpoint-69000`.
Scorers: `analysis/score_frames_psnr.py` (PSNR/SSIM + time-reversal) and
`analysis/score_control_flanked.py` (raw motion correlation). Chain:
`scripts/run_camera_shake_evals.sh`.

| dataset | arm | PSNR | SSIM | reversed | motion corr (raw) |
|---|---|---|---|---|---|
| Adobe240 | no dynamics | 20.25 | 0.653 | 5/20 | 0.224 |
| Adobe240 | Stage C | 22.35 | 0.708 | 0/20 | 0.441 |
| Adobe240 | oracle | 22.91 | 0.722 | 0/20 | 0.492 |
| GoPro | no dynamics | 19.33 | 0.560 | 3/20 | 0.216 |
| GoPro | Stage C | 21.13 | 0.620 | 0/20 | 0.445 |
| GoPro | oracle | 21.47 | 0.632 | 0/20 | 0.475 |

| comparison | Adobe240 PSNR | GoPro PSNR | Adobe240 corr | GoPro corr |
|---|---|---|---|---|
| no-dyn -> oracle | +2.65, t=9.02 | +2.14, t=7.62 | +0.268, t=9.14 | +0.260, t=9.79 |
| no-dyn -> Stage C | +2.09, t=8.04 | +1.79, t=6.19 | +0.217, t=7.63 | +0.230, t=8.48 |
| **oracle -> Stage C** | **-0.56, t=-4.62** | **-0.35, t=-3.31** | **-0.052, t=-4.21** | **-0.030, t=-3.23** |

1. **Dynamics does the work on camera-shake too** (t 6-12 on every measure), as on FMO.
2. **Time reversal**: 0/20 with dynamics on both datasets; without dynamics 5/20
   (Adobe240) and 3/20 (GoPro). The dynamics tokens are what fixes direction.
3. **The predictor gap is REAL and significant here** (Stage C below oracle on both
   datasets, PSNR and correlation). On FMO at 69k it was not significant
   (t=-1.64) but that arm had the least statistical power (see the 2026-09-24
   entry: the point estimate never moved). Stage C recovers ~79% (Adobe240) and
   ~84% (GoPro) of the oracle's PSNR benefit, consistent with ~85-89% on FMO.
   **Stop describing Stage C as parity with the oracle.**
4. **Absolute PSNR (21-23 dB) is NOT comparable to the older 29.94 dB hires
   number.** These are computed from the saved mp4s (both GT and prediction are
   lossy-compressed) with no crop, whereas the old figure came from the trainer's
   in-memory metric. Relative comparisons between arms are valid; do not read the
   absolute values as a regression.

## 2026-09-25 -- Burst sweep at 736 REPLICATES the 512 result

Same design at the reporting resolution (5 modules on every arm; energy ratios
0.79/0.96/0.93). Raw correlation, n=18-20:

| S | full length | common 9-frame window |
|---|---|---|
| 1 | 0.399 | 0.399 |
| 3 | 0.423-0.432 | 0.425 |
| 5 | 0.295-0.304 | 0.450 |

S3 -> S5 at full length: -0.127, t=-2.40, p=0.021; on the common window it flips to
+0.025, t=0.74, p=0.61. S1 -> S5 on the common window: +0.051, t=1.27, p=0.14.
Same conclusion as 512: the S=5 full-length drop is a LENGTH effect; extra blur
frames do not hurt (they trend slightly positive, not significantly). Two
resolutions agreeing raises confidence that this is not a resolution artifact.

## 2026-09-25 -- Scene-clustered re-scoring (PARTIAL: 6 of 24 comparisons, FMO/TbD only)

The clip-level paired tests treat clips as independent, but clips within a scene are
overlapping consecutive bursts. `analysis/cluster_stats.py` adds a scene-mean t-test
(df = scenes-1) and a scene bootstrap; both scorers now print a `CLUSTERED` line.
FMO/TbD has only **4 scenes**, so df=3 (very low power) and the bootstrap CI is too
narrow (few clusters) -- read the t-test p and the scenes-up count, not the CI.
Job `rescore_cluster` was cancelled after 6 of 24 finished; results in
`/scratch/gilbreth/wweligam/rescore_cluster/`.

| comparison (raw corr unless noted) | clip-level | scene-level (4 scenes) | scenes up |
|---|---|---|---|
| ck69k no-dyn -> oracle | +0.319, t=11.37 | t=9.81, p=0.002 | 4/4 |
| ck69k no-dyn -> Stage C | +0.283, t=8.44 | t=6.33, p=0.008 | 4/4 |
| **ck69k oracle -> Stage C** | -0.036, t=-1.64 | **t=-1.81, p=0.168** | 1/4 |
| ck69k oracle -> Stage C, PSNR | -0.232, t=-1.19 | t=-1.49, p=0.233 | 1/4 |
| ck58k no-dyn -> oracle | +0.216, t=5.28 | t=4.49, p=0.021 | 4/4 |
| **ck58k oracle -> Stage C** | -0.033, t=-3.11 | **t=-2.37, p=0.098** | 0/4 |

* **Dynamics-does-the-work SURVIVES clustering** on FMO/TbD (p 0.002-0.021, every scene
  improves), though the 58k margin is now only p=0.021.
* **The FMO/TbD predictor gap is NOT significant at scene level** at either checkpoint
  (p=0.098 at 58k, p=0.168 at 69k). The 58k "significant, p=0.005" figure in the
  2026-09-24 entry was a clip-level artifact and is downgraded. The DIRECTION is
  consistent (Stage C below oracle in 4/4 scenes at 58k, 3/4 at 69k) but 4 scenes
  cannot establish it.
* NOT yet rescored (18 comparisons): Adobe240/GoPro (8 and 11 scenes -- the datasets
  with real power for the predictor gap), the 320-resolution architecture claim
  (BASE140k / FT149500 / XATTN19500), 512-vs-320, and both burst sweeps.

## 2026-09-25 -- Scene-clustered re-scoring COMPLETE (24/24), supersedes the partial entry above

Job `rescore_cluster` (standby a30, 4 CPUs, 18 min) finished. Unit of replication = SCENE
(FMO/TbD 4 scenes, Adobe240 8, GoPro 11). Raw correlation unless noted; p = scene-mean t-test.

| claim | clip-level | scene-level | scenes up | verdict |
|---|---|---|---|---|
| **cross-attention > PGA (BASE140k -> XATTN19500), 320** | +0.115, t=3.42 | t=13.6, **p=0.001** | 4/4 | HOLDS |
| cross-attention > data-only FT | +0.111, t=3.08 | t=3.44, p=0.041 | 4/4 | HOLDS (marginal) |
| data-only FT vs baseline | +0.004, t=0.14 | p=0.90 | 2/4 | still NOTHING |
| **512 vs 320** (31k vs 29k, same lineage) | +0.073, t=1.66 | t=2.93, **p=0.061** | **4/4** | suggestive, NOT established |
| dynamics does the work, FMO @69k | +0.319 | p=0.002 | 4/4 | HOLDS |
| dynamics does the work, Adobe240 | +0.268 | p<0.001 | 8/8 | HOLDS |
| dynamics does the work, GoPro | +0.260 | p<0.001 | 11/11 | HOLDS |
| **oracle -> Stage C, Adobe240** | -0.052, t=-4.21 | **p=0.019** (PSNR p=0.008) | 0/8 | REAL gap |
| **oracle -> Stage C, GoPro** | -0.030, t=-3.23 | **p=0.027** (PSNR p=0.046) | 2/11 | REAL gap |
| oracle -> Stage C, FMO/TbD @69k | -0.036, t=-1.64 | p=0.168 | 1/4 | not established |
| oracle -> Stage C, FMO/TbD @58k | -0.033, t=-3.11 | p=0.098 | 0/4 | not established |
| oracle -> Stage C, FMO/TbD @37k (512) | +0.009 | p=0.825 | 3/4 | null |
| 58k -> 69k, oracle / Stage C (FMO) | +0.049 / +0.047 | p=0.171 / 0.266 | 3/4 each | not established |

* Clustering did NOT kill the headline results; it removed one overstated one (FMO predictor
  gap at 58k, clip p=0.005 -> scene p=0.098) and confirmed the camera-shake predictor gap,
  which has enough scenes (8, 11) to be tested properly.
* The cross-attention result got STRONGER under clustering (t=13.6): the effect is very
  consistent across scenes even though clips within a scene are correlated.
* **512 vs 320 moved from "p~0.1" to p=0.061 with 4/4 scenes up** -- the best evidence yet
  that 512 helps, still under the .05 line and still confounded (2000 extra steps).
* FMO/TbD has only 4 scenes: df=3, and the scene bootstrap CI is unreliable at that size
  (use the t-test p and scenes-up). The predictor gap on FMO cannot be resolved without more
  scenes (T0c: a bigger held-out set).

**Burst sweep (Stage C path, ckpt-69000), 4 scenes, raw corr:**

| | 512 full length | 512 common 9f | 736 full length | 736 common 9f |
|---|---|---|---|---|
| S1 -> S3 | +0.036, p=0.44 | +0.037, p=0.54 | +0.032, p=0.71 | +0.026, p=0.62 |
| S3 -> S5 | **-0.152, p=0.039** | -0.020, p=0.81 | -0.127, p=0.16 | +0.025, p=0.60 |
| S1 -> S5 | **-0.115, p=0.024** | +0.018, p=0.20 | -0.095, p=0.24 | +0.051, p=0.52 |

The S=5 full-length drop is significant only at 512 and vanishes on the common 9-frame window
at both resolutions (all common-window p >= 0.20): a length effect, not a burst-context effect.
Only 4 scenes, so "no significant burst effect" means "none detectable", not "equivalent".

Files: `/scratch/gilbreth/wweligam/rescore_cluster/*.txt`; scripts `scripts/sbatch_rescore.sh`,
`analysis/cluster_stats.py`.

## 2026-09-25 -- RETRACTED: the seed-floor "noise floor of exactly 0" was a bug (seed overwritten on resume)

The first T0b run (oracle + Stage C, config `seed` 1-4 + the seed-42 arm) gave a seed-to-seed SD of
**0.0000** on every metric. Not physical: the same clip's generated video had an **identical md5
across all five seeds**. The config seed did reach each run (`'seed': 1` ... `'seed': 4` in the log),
but every eval RESUMES from a checkpoint, and `accelerator.load_state()` then restores the
checkpoint's saved RNG state (`random_states_0.pkl`), overwriting the seed set at startup.
**All earlier evals therefore drew the same noise stream** (and oracle and Stage C shared it per clip,
so their paired gap is less noisy than independent draws would be -- the real noise floor of the
predictor gap is unknown until the rerun below).

Fix: new config key `sampler_seed` (`Wan22VJEPADynamics_model.py`), a `torch.Generator` seeded per
(seed, clip index) and applied AFTER the resume; unset = old behaviour unchanged. Verified with two
2-clip runs of identical config seed 42 and `sampler_seed` 11 vs 12: md5 now differs on both clips.
Invalid outputs quarantined as `X_INVALID_identical_*`, `seedfloor_done_INVALID_identical_videos`,
`seed_floor_INVALID_identical_videos.txt`. Rerun of all 8 arms launched on k048 (fixed verifier).

Lesson: a control that the varied factor actually varies (hash two outputs) takes minutes; skipping
it cost 3.5 GPU-hours. Also: `seed:` in these configs does NOT randomise anything on a resumed eval.

## 2026-09-26 -- Sampler-noise floor (VALID rerun): single-run differences below ~0.05-0.08 in raw correlation are not resolvable

Oracle and Stage C, FMO/TbD 736, frozen ckpt-69000 / Stage B ckpt-20000, same 28 clips, 5 noise
draws (4 x `sampler_seed` + the original checkpoint-RNG arm). Seeds now differ (per-seed means
differ), so the earlier identical-video bug is fixed. `analysis/seed_floor.py`.

| | per-seed means | mean | SD across seeds | per-clip seed SD |
|---|---|---|---|---|
| oracle, raw corr | .4756 .5014 .4469 .4867 .4467 | 0.4715 | **0.024** | 0.122 |
| Stage C, raw corr | .4393 .4329 .3596 .4561 .3879 | 0.4152 | **0.040** | 0.141 |
| **gap Stage C - oracle, corr** | -.036 -.069 -.087 -.031 -.059 | **-0.056** | **0.023** | |
| oracle, PSNR (dB) | 33.55 33.74 33.47 33.76 33.60 | 33.62 | 0.128 | 0.74 |
| Stage C, PSNR (dB) | 33.31 33.46 32.96 33.63 33.30 | 33.33 | 0.245 | 0.83 |
| **gap Stage C - oracle, PSNR** | -.23 -.28 -.50 -.14 -.30 | **-0.29 dB** | 0.134 | |

1. **The predictor gap is real with respect to sampler noise on FMO/TbD**: negative in all 5 draws
   (corr: 5/5, PSNR: 5/5), seed-averaged -0.056 corr / -0.29 dB, about 2.2-2.4 single-run SDs and
   5 standard errors from zero across seeds. The single-run values reported earlier (-0.033 at
   58k, -0.036 at 69k, -0.23 dB) sat at the small end of the noise; the better estimate is about
   -0.056 +- 0.010. (The scene-level test, p=0.10-0.17, is limited by having only 4 scenes,
   not by sampler noise.)
2. **A single run's mean correlation carries +-0.024 (oracle) / +-0.040 (Stage C) of sampler
   noise** (per-clip SD ~0.12-0.14). Differences of the size we have been reading -- 58k->69k
   (+0.047/+0.049), the burst-size deltas (<=0.05), and 512 vs 320 (+0.073) -- are within about
   1-2 single-run SDs. They were correctly called "not established"; 512-vs-320 in particular is
   weaker than its scene p=0.061 implies. Resolving deltas this small needs several noise draws
   averaged per arm.
3. **Same-noise pairing helps a lot**: the gap SD (0.023) is about half of what independent noise
   would give (sqrt(0.024^2+0.040^2)=0.047), because oracle and Stage C share the same noise for
   a given seed and clip. Paired comparisons on shared noise are far less noisy than unpaired ones.
4. Caveat: 5 draws, one dataset (FMO/TbD), one checkpoint; the seed-42 arm uses a different
   noise mechanism (checkpoint-restored RNG) but is an independent draw.

## 2026-09-26 — P1a token-intervention suite (oracle tokens, ckpt-69000 frozen, 512, 28 clips / 4 scenes FMO-TbD, 2 sampler seeds averaged)
Source: /scratch/gilbreth/wweligam/p1a_scores.txt (run SUITE_P1a_tokens_2026_09_26_03_29_36; exit 0, 5 modules loaded, 0 tracebacks, all 24 arm-seed dirs DONE; scorer flagged no arm as md5-identical to `none`).
Positive control `none` corr 0.477; floor `zero` 0.146 (-0.332, scene p=0.009, 27/28 clips down).
- Temporal manipulations do NOT matter: reverse_within_seg +0.007, shift_segments +0.004, shuffle_tubelets +0.003 (all within the ±0.024 single-run sampler noise; retained ≈1.00). Spatial-cell shuffle also ~0 (+0.002) and noise1 -0.004.
- keeping only one segment: seg0 1.00 retained, seg1 0.91, seg2 0.83 (n.s.; 1 of 3 segments suffices).
- swap_donor (tokens from a different scene): corr with OWN GT drops to 0.278 (retained 0.40, scene p=0.032); corr with the DONOR's GT motion is ≈0 (-0.002), donor>own in 1/28 clips. The output does NOT follow the donor's motion.
Interpretation (tentative): the cross-attention tokens are used as a coarse per-clip cue (content/scene-specific: the wrong clip's tokens hurt by ~60% of the token benefit), NOT as a temporally ordered motion code: shuffling/shifting their time order is invisible to the metric. (The token-axis `reverse_time` arm originally listed here was removed on 2026-10-04: flipping V-JEPA's token time axis is not equivalent to the tokens of a reversed video, see the entry below; its result is no longer reported.) This weakens the claim "gain comes from temporal information in the tokens" (C1) and the claim of controllable motion via tokens. Caveats: only 4 scenes (scene p has little power), single metric (moving-content correlation), 2 seeds; the metric may be insensitive to direction (|gray-median| map is sign-free, PSNR-reversal rate is ~1/28 for all arms), so "order does not matter" is for THIS metric. Donor test compares against the donor's GT at the donor's location, so ≈0 is expected even for a partially donor-following output; a direction-sensitive metric (optical-flow direction) is needed before claiming "does not follow". Pending: hires confirmation, more scenes.

## 2026-09-26 — P1b module knockouts (same setup as P1a: oracle, ckpt-69000 frozen, 28 clips / 4 scenes, 2 seeds)
Source: /scratch/gilbreth/wweligam/p1b_scores.txt (exit 0, 5 modules, 0 tracebacks, 12 arm-seed dirs). Positive control `none` = 0.477, reproduces P1a's number exactly.
- xattn_off (cross-attention output disabled): 0.097 (-0.380, 28/28 clips down, scene p=0.004, energy ratio 4.75 = degenerate output). Below the P1a `zero`-token floor (0.146).
- lora_off (Wan LoRA adapters disabled, cross-attn kept): 0.103 (-0.374, 28/28, p=0.007, energy 4.42). Cross-attention alone without the co-adapted LoRA is as bad as no cross-attention: the two are jointly required.
- pos_t_shuffle +0.005, pos_s_shuffle +0.002, pos_zero +0.011 (p=0.053; all inside ±0.024 noise): the learned positional embeddings on the token keys are NOT needed, consistent with P1a (tokens' temporal/spatial arrangement is not exploited).
Caveats: the knockout arms produce out-of-distribution activations, so their collapse (energy 4-5x) shows the model depends on those modules, not that they carry "temporal" information; 4 scenes; one metric. Together P1a+P1b: gain needs the xattn+LoRA pathway and the token CONTENT, but not token order/position.

## 2026-09-26 -- COMMON test set for all baselines and our methods (protocol + status)
Problem (measured): each method had been run on a different clip list -- ours vs Jin2018 share 1/20 (Adobe240) and 0/20 (GoPro) clips; Tedla used 8 unscored GoPro val images and 6 peak-motion falling-object clips.
Protocol (user-approved, S=1): ONE blur (11-frame window at the clip centre) -> 7 GT frames (centred 9-frame window, evenly subsampled). Multi-blur (S>1) comparisons are later: baselines run one pass per blur and stitch.
Set: gopro 20 clips / adobe 20 clips (linspace over test indices, 1280x736; = the clips Jin2018 was already run on -- verified stem-identical), fmo_tbd 36 clips (4 scenes x 9 stride-8 centres in peak+-32, motion_square 736). Manifests: findings/common_testset/manifest_*.json; inputs/GT/floor-control: /scratch/gilbreth/wweligam/common_testset/<ds>/ (blur PNGs + 7-frame GT mp4 + blur-x7 control). Sanity: FMO blur vs mean of GT frames MAE 1.5-2.7/255 (consistent).
Caveat: GoPro/Adobe = 20 evenly spaced indices (GoPro 20 clips over ~20 scenes? see manifest n_scenes; few clips per scene => cluster stats via scene_of). Jin2018 falling-objects must be re-run on the 36 clips; Tedla/Zhong/HyperCUT and ours (S=1) still to run -- queued behind the P2 pipeline on k048.
Update (user: 10 falling objects enough, diverse scenes): fmo_tbd common set reduced to 10 clips = 4 held-out scenes (only 4 exist; the other 12 FMO/TbD scenes are training data) x 2-3 clips spread across each scene's motion range (v_key 3, v_marker 3, out2 2, outf1 2). Full 36 kept as manifest_fmo_tbd_full36.json / common_testset/fmo_tbd_full36/. GoPro/Adobe240 (20 each) are the priority datasets.

## 2026-09-27 -- P2a (trained no-dynamics baseline) vs `ref`: the 8000-step from-scratch recipe did NOT learn to use the tokens -> P2 is uninformative as designed
Source: /scratch/gilbreth/wweligam/p2_scores_ref_nodyn.txt (68 clips = 28 FMO/TbD + 20 Adobe240 + 20 GoPro, 23 scenes, 2 noise seeds averaged; both runs verified: 8000 steps, round-trip ok, 5 modules, 68/68 clips).
| | corr ref | corr nodyn | d | scene p | PSNR ref | PSNR nodyn | d | scene p |
|---|---|---|---|---|---|---|---|---|
| ALL | 0.229 | 0.228 | -0.002 | 0.55 | 24.14 | 25.04 | +0.90 | 0.010 |
| FMO/TbD | 0.177 | 0.157 | -0.020 | 0.12 | 31.67 | 32.38 | +0.70 | 0.10 |
| Adobe240 | 0.279 | 0.285 | +0.006 | 0.69 | 19.90 | 20.85 | +0.95 | 0.05 |
| GoPro | 0.253 | 0.269 | +0.016 | 0.49 | 17.82 | 18.95 | +1.13 | 0.14 |
Reading: with dynamics ON, an 8000-step from-scratch run is no better than dynamics OFF on motion correlation and is WORSE in PSNR. For scale, the 69000-step Stage A oracle scored 0.477 (FMO/TbD, 512) vs 0.146 with zeroed tokens (P1a) -- here `ref` on FMO/TbD is 0.177, i.e. it has not learned to use the tokens yet. So this run does not answer the fairness question ("is the gain just extra training / LoRA?"); it only shows that 8000 steps from scratch is far too short for the xattn pathway to switch on.
Consequences: P2b (naive burst) and P2c (DINOv2) trained with the same recipe will be compared against a `ref` that has no dynamics benefit, so they cannot rank encoders/conditioning either. Needed before spending the remaining ~15 GPU-hours on them: (a) a longer budget or a warm start (e.g. from a checkpoint where the xattn has already engaged), or (b) evidence that `ref` is still improving at 8000 (its val loss / a mid-run checkpoint eval). Caveats: absolute correlations are on different clip sets/resolution (512, 68 clips) than the 0.477; per-seed noise +-0.02-0.04; scene p has little power.

## 2026-09-27 -- common-set brightness difference: not a bug, user-confirmed by eye
Our S=1 common-set outputs and Jin2018's outputs measured 2.7-5.5x brighter (mean pixel value) than GT on GoPro/Adobe240/falling-objects clips (e.g. GoPro clip: GT mean 37/255, ours 125/255). Initially flagged as a possible pipeline bug since it did not match the validated S=3 dynamic-burst runs (which match GT brightness almost exactly, e.g. 78.2 vs 79.9). User reviewed the actual videos directly and reports they "look fine" -- treating this as the model's own exposure/lighting behavior at S=1, not a defect, and NOT blocking further common-set work on it.
Consequence for scoring: PSNR is NOT comparable across methods with different exposure (jin2018/ours vs tedla/control) -- do not read PSNR gaps as quality differences until/unless exposure is normalized. Motion correlation (|gray-temporal median| deviation) is largely brightness-invariant and remains the primary metric for the common-set comparison. Common-set scores already computed (see /scratch/gilbreth/wweligam/common_scores.txt, 2026-09-27): GoPro/Adobe240 corr ordering ours_oracle > ours_stagec > tedla > ours_nodyn > jin2018 > control; falling objects (10 clips) ours_oracle ~ ours_stagec > jin2018/tedla ~ control -- not yet written up with scene-level stats or caveats in a dedicated section.

## 2026-09-27 -- RETRACTION: the common-set brightness issue was a bug in the GT export, not in our model
The earlier entry ("common-set brightness difference: not a bug, user-confirmed by eye") was wrong about WHERE the bug was.
User asked to check the GT value range directly. `analysis/make_common_testset.py` assumed `sample["blur"]` and
`sample["wan_frames"]` were in [0,1] for GoPro/Adobe240 and wrote them through `u8()` (`.clip(0,1)*255`) unconverted.
Both are actually pm1 ([-1,1]) for GoProSequenceDataset/Adobe240Dataset (confirmed directly:
`blur range -0.97 1.0`, `wan_frames range -1.0 1.0`). The clip() therefore zeroed every pixel below mid-gray and
compressed the rest, corrupting BOTH the GT videos used for scoring AND the blur PNGs fed to Jin2018/Tedla/our own
model's inference. (FallingObjectsDataset's export was fine because `wan_frames_range="01"` was already set for it.)
Verified: a GoPro GT frame rendered at full resolution before the fix looked like a night scene (mean 37/255); after
fixing `make_common_testset.py` (added `wan_frames_range: "01"` to BASE; blur now unconditionally `(x+1)/2`; kept
falling-objects, which is unaffected) the SAME frame is a normal daylight street scene (mean 126/255) and visually
matches what our model had been outputting all along. **Our model's S=1 output was correct the whole time** -- the
"2.7-5.5x too bright" measurement compared a correct prediction to a corrupted GT. Jin2018's blur INPUT was also
corrupted for gopro/adobe (not just its comparison), so its earlier common-set numbers are invalid, not just uncomparable.
Fixed and rebuilt: findings/common_testset/manifest_{gopro,adobe}.json + common_testset/{gopro,adobe}/{gt_*.mp4,blur_inputs/}
(fmo_tbd untouched, already correct). Jin2018 rerun on the corrected blur_inputs (common_testset/{gopro,adobe}/jin2018_v2/,
verified 20/20 each). Tedla rerun in progress (common_testset/{gopro,adobe}/tedla_v2/). Our model does NOT need
rerunning (it reads the dataset directly, bypassing this export path) but must be RE-SCORED against the corrected GT --
the earlier common_scores.txt table is invalid and superseded once Tedla's rerun finishes.
Lesson: when in doubt about a pixel value's range/convention, check it directly against the dataset source instead of
trusting a code comment -- the exact same class of silent-wrong-number failure as [[fmo_gt_bbox_not_frame_aligned]] and
[[dictasmember_silent_empty_dict]]: nothing raised an error, the video just looked plausible (a dusk photo) instead of wrong.

## 2026-09-27 -- CORRECTED common-set scores (post pm1/01 bugfix), all methods on identical clips
Same protocol as before (S=1, single 11-frame blur -> 7-frame GT), now on the corrected GT/blur exports for
gopro/adobe (fmo_tbd unaffected). Jin2018 and Tedla rerun on the corrected blur_inputs; ours did not need
rerunning (reads the dataset directly). Source: /scratch/gilbreth/wweligam/common_scores_v2.txt.

| dataset | method | corr | d-vs-control | PSNR |
|---|---|---|---|---|
| GoPro (20 clips/20 scenes) | control (floor) | -0.003 | - | 23.45 |
| | jin2018 | 0.280 | +0.283 | 18.25 |
| | tedla | 0.245 | +0.248 | 20.34 |
| | ours nodyn | 0.264 | +0.267 | 22.67 |
| | ours Stage C | 0.429 | +0.432 | 23.38 |
| | ours oracle | 0.482 | +0.485 | 23.97 |
| Adobe240 (20/20) | control | -0.006 | - | 26.24 |
| | jin2018 | 0.257 | +0.263 | 18.98 |
| | tedla | 0.335 | +0.341 | 22.94 |
| | ours nodyn | 0.296 | +0.302 | 25.08 |
| | ours Stage C | 0.393 | +0.399 | 25.30 |
| | ours oracle | 0.451 | +0.457 | 26.36 |
| FMO/TbD (10/10, unaffected by the bug) | control | -0.003 | - | 35.82 |
| | jin2018 | 0.019 | +0.022 (n.s., p=0.195) | 25.27 |
| | tedla | 0.098 | +0.100 (p=0.035) | 30.84 |
| | ours nodyn | 0.196 | +0.198 (p=0.004) | 33.65 |
| | ours Stage C | 0.426 | +0.428 (p<0.001) | 34.75 |
| | ours oracle | 0.431 | +0.434 (p<0.001) | 34.59 |

Reading: our Stage C and oracle beat both external baselines and our own no-dynamics arm on every dataset, on
both metrics, now that PSNR is comparable across methods (18-27dB throughout, not the ~9dB seen with the
corrupted GT). Stage C is close behind oracle everywhere (predictor gap holds up on the common set too).
On GoPro/Adobe240 tedla/jin2018 land between control and our nodyn arm; on falling objects both external
baselines are much closer to the control floor than to any of our arms (jin2018 not even significant, p=0.195).
Caveats: 20 scenes for GoPro/Adobe240 = 1 clip/scene (scene-level test == clip-level test there, no real
scene replication -- treat those scene_p values as clip-level, not independently validated across scenes);
10 scenes for falling objects is a genuine deliberate subsample (user: 10 diverse clips is enough, see the
2026-09-26 fmo_tbd reduction entry); PSNR crop convention not applied here (whole-frame); this is the S=1
single-blur protocol only, not the burst S=3 regime the main results use.

## 2026-09-27 -- P2 at 30000 steps: `ref` now clearly beats the matched no-token baseline (the 8000-step null was undertraining)
Both arms continued in place from their step-8000 checkpoints to 30000 (same recipe: LoRA r8, xattn rank 64, lr 1e-4, bs2, burst 3, oracle V-JEPA tokens for `ref`; tokens disabled from step 0 for `nodyn`), evaluated on the same 68 clips x 2 sampler seeds (512), frozen checkpoints, both evals verified (5 modules, 68/68, videos differ across arms).
Source: /scratch/gilbreth/wweligam/p2_scores_30k.txt
| | corr ref30k | corr nodyn30k | d | scene p | PSNR ref30k | PSNR nodyn30k | d | scene p |
|---|---|---|---|---|---|---|---|---|
| ALL (68 clips, 23 scenes) | 0.375 | 0.249 | -0.126 | <0.001 | 25.82 | 24.54 | -1.28 | <0.001 |
| Adobe240 (20/8 scenes) | 0.441 | 0.303 | -0.138 | 0.018 | 22.16 | 20.35 | -1.81 | 0.020 |
| GoPro (20/11) | 0.409 | 0.258 | -0.151 | 0.001 | 20.58 | 18.19 | -2.40 | <0.001 |
| FMO/TbD (28/4) | 0.303 | 0.205 | -0.099 | 0.115 (n.s.; 4 scenes) | 32.17 | 32.06 | -0.11 | 0.57 |
Trajectory: at 8000 steps ref 0.229 vs nodyn 0.228 (identical, tokens not engaged); at 30000 ref 0.375 vs nodyn 0.249. `ref` gained +0.146, `nodyn` only +0.021.
Reading: the tokens engage between 8k and 30k steps; once they do, the token-conditioned model beats an identically-trained no-token model by 0.126 correlation (nodyn better on 8/68 clips only; GoPro 0/20, Adobe 1/20) and 1.3 dB. This answers the fairness question P2a was built for: the Stage A gain is not "extra LoRA training", a no-token model trained for the same steps from the same start stays far behind. The user's diagnosis (ref needed more steps) was right.
Caveats: single training run per arm (no training-seed replication); 2 sampler draws averaged; FMO/TbD has only 4 scenes (p=0.115 is power-limited, direction agrees); `ref` uses ORACLE tokens (upper bound), not Stage C; absolute ref30k FMO corr (0.303) is still below the 69000-step Stage A oracle (0.477 on a different 28-clip set at 512), so ref is not fully converged at 30k; P2b (naive burst) and P2c (DINOv2) not yet run at this budget.

## 2026-09-29 -- P2c (DINOv2 encoder swap) at 30000 steps: DINOv2 gives NO benefit over no dynamics at all
Same matched-budget protocol as P2a/P2b: continued from checkpoint-8000 in place to 30000 (dynamics_encoder_type: dinov2, same token slot/cross-attention pathway as ref/nodyn), evaluated on the same 68 clips x 2 seeds. Verified (5 modules, 68/68 clips, seeds differ by md5).
Source: /scratch/gilbreth/wweligam/p2_scores_dino30k.txt
| | corr ref30k | corr nodyn30k | corr dino30k | PSNR ref30k | PSNR nodyn30k | PSNR dino30k |
|---|---|---|---|---|---|---|
| ALL (68 clips, 23 scenes) | 0.375 | 0.249 | 0.244 | 25.82 | 24.54 | 24.44 |
| Adobe240 (20/8) | 0.441 | 0.303 | 0.290 | 22.16 | 20.35 | 20.12 |
| GoPro (20/11) | 0.409 | 0.258 | 0.248 | 20.58 | 18.19 | 18.02 |
| FMO/TbD (28/4) | 0.303 | 0.205 | 0.208 | 32.17 | 32.06 | 32.11 |
d(dino30k vs ref30k) is -0.131 (ALL), essentially the SAME as d(nodyn30k vs ref30k) = -0.126 -- dino30k is NOT distinguishable from the no-dynamics floor on any dataset or metric (all differences between nodyn30k and dino30k are within noise: e.g. Adobe240 corr 0.303 vs 0.290, GoPro 0.258 vs 0.248).
Reading: swapping V-JEPA for DINOv2 in the exact same cross-attention slot recovers essentially NONE of the oracle-token benefit. Since DINOv2 is a strong general-purpose visual encoder (just not temporally pretrained -- per-frame features, pair-averaged to fake a tubelet), this points at V-JEPA's TEMPORAL/video pretraining specifically being what the tokens need, not merely "some visual encoder in that slot." Consistent with the P1a finding that token content (not order) carries the signal -- DINOv2's per-frame-averaged tokens apparently don't carry usable motion content the way V-JEPA's do, even though both are fed through the identical pathway.
Caveats: single training run per arm; DINOv2 tokens were pair-averaged only to match V-JEPA's tubelet grid shape, not because DINOv2 has any temporal structure of its own -- a fairer DINOv2 comparison might use a genuinely video-native alternative (e.g. VideoMAE) instead, which was in the original plan but not run.

## 2026-09-29 -- P2b (naive burst-blur conditioning) at 30000 steps: BEATS the V-JEPA oracle at matched budget -- important, but NOT a clean "V-JEPA is useless" result
Same matched-budget protocol as P2a/P2c: continued from checkpoint-8000 in place to 30000
(disable_dynamics_injection: true -- no V-JEPA tokens/cross-attention at all -- PLUS
burst_blur_conditioning: true, the 3 raw blur images laid out in pixel time and VAE-encoded into
tokens added at block 0; see WanDynamicsCrossAttn vs WanBlurCondEmbed in Wan22VJEPADynamics_model.py).
Evaluated on the same 68 clips x 2 seeds. Verified (5 modules, 68/68 clips, seeds differ by md5).
Source: /scratch/gilbreth/wweligam/p2_scores_naiveburst30k.txt
| | corr ref30k | corr naiveburst30k | d | PSNR ref30k | PSNR naiveburst30k | d |
|---|---|---|---|---|---|---|
| ALL (68 clips, 23 scenes) | 0.375 | **0.633** | **+0.258, p<0.001** | 25.82 | **28.37** | **+2.55, p<0.001** |
| Adobe240 (20/8) | 0.441 | 0.706 | +0.265, p<0.001 (20/20 up) | 22.16 | 25.55 | +3.38 |
| GoPro (20/11) | 0.409 | 0.726 | +0.318, p<0.001 (20/20 up) | 20.58 | 24.48 | +3.89 |
| FMO/TbD (28/4) | 0.303 | 0.514 | +0.211, p=0.018 (22/28 up) | 32.17 | 33.18 | +1.01 |
Energy ratio 1.22 overall (1.62 FMO) vs ref's 1.12 (1.43 FMO) -- somewhat over-active but nowhere
near the "zero"/no-dynamics floor's degenerate ~2-4.75 seen in P1a/P1b, so this is not hallucinated
noise being mistaken for motion.

**Read this carefully -- it is NOT evidence that V-JEPA/dynamics tokens don't matter, for two reasons:**
1. This is a TWO-variable swap (dynamics OFF + burst-blur-conditioning ON), not the single-variable
   "remove V-JEPA, change nothing else" test P2a (nodyn30k) already ran -- and nodyn30k (dynamics off,
   no blur conditioning) is BELOW ref30k (0.249 vs 0.375, the original P2a finding). So the gain here
   is not "V-JEPA hurts"; it is "directly feeding the raw blur pixels as dense VAE tokens is a
   stronger conditioning MECHANISM than a 12-tubelet cross-attention summary from V-JEPA, at this
   task and this budget." A confound worth naming, not resolved here: burst_blur_conditioning tokens
   carry far more raw information (a whole encoded video) than the compressed V-JEPA grid.
2. Matched-budget (30000 steps from scratch) is NOT the deployed regime. The actual Stage A
   checkpoint-69000 (V-JEPA + cross-attention) was trained far longer (69000+ steps) and is the
   basis for every other reported number in this project (Stage C, the common-set comparison, etc.).
   Whether a burst-blur-conditioned model trained to that SAME 69000-step budget would beat the
   deployed checkpoint is an open, unanswered question -- this result is at 30k only.

**What this DOES establish:** the P2b fairness question from the original plan ("is naive burst
conditioning a strong alternative?") -- yes, clearly, at matched budget it is the strongest arm of
the whole P2 family by a wide margin, ahead of dino30k, nodyn30k, and even ref30k (oracle V-JEPA).
This is a genuine, actionable finding: burst-blur pixel conditioning deserves a serious look as
either a replacement or a complement to the V-JEPA cross-attention pathway, not just a fairness
control to be dismissed.
Caveats: single training run per arm; naiveburst30k likely benefits from the STRAIGHTFORWARD
correspondence between conditioning-image-in-time and target-video-in-time (an inductive bias
cross-attention doesn't have); reversal rate is 0/68 (as good as ref); not yet tested at the 69000-
step deployed budget, nor against Stage C (the trained predictor) or on the common test set.

## 2026-09-29 -- P2g (joint predictor fine-tune, +3000 steps from ckpt-69000): narrows the Stage C gap, but a no-predictor-training control narrows it almost as much
Two arms, both warm-started from the frozen checkpoint-69000 (+ Stage B predictor ckpt-20000) for 3000 more steps: `joint` (predictor LoRA trainable, aux loss 0.1 to the oracle tokens, lr 3e-5) and `control` (identical budget, predictor frozen, no aux loss -- isolates "just more Wan-side steps" from "actually training the predictor"). Each evaluated at Stage C and oracle on the same 68 clips x 2 seeds as the P2 family; `baseline` = the original checkpoint-69000/Stage-B-20000 pair, unevaluated. All 6 evals verified (5 modules, 68/68 clips).
Source: /scratch/gilbreth/wweligam/p2g_scores.txt. Gap = oracle corr - Stage C corr (smaller = predictor closer to oracle):
| | baseline oracle | baseline stagec | gap | joint oracle | joint stagec | gap | control oracle | control stagec | gap |
|---|---|---|---|---|---|---|---|---|---|
| ALL (68) | 0.455 | 0.418 | 0.037 | 0.502 | 0.489 | **0.013** | 0.491 | 0.472 | 0.019 |
| Adobe240 (20) | 0.490 | 0.433 | 0.057 | 0.518 | 0.506 | **0.012** | 0.535 | 0.503 | 0.032 |
| GoPro (20) | 0.450 | 0.409 | 0.041 | 0.529 | 0.517 | **0.012** | 0.531 | 0.508 | 0.023 |
| FMO/TbD (28) | 0.434 | 0.413 | 0.021 | 0.471 | 0.457 | 0.014 | 0.431 | 0.423 | **0.008** |
Both joint AND control shrink the gap vs baseline everywhere (joint: 65% smaller on ALL; control: 49% smaller). Joint has the smallest gap on Adobe240/GoPro; control has the smallest on FMO/TbD. **Neither arm is a clean win for "training the predictor jointly helps"** -- control, which never touches the predictor at all, closes most of the same gap just from 3000 more Wan-side steps continuing from ckpt-69000. The oracle arms ALSO both improved substantially over baseline_oracle (e.g. GoPro 0.450 -> 0.529/0.531), meaning most of what's being measured here is "the Wan side keeps improving with more training," not "the predictor caught up to the oracle."
Reading: this experiment does not cleanly separate "does joint fine-tuning close the predictor gap" from "does more training close it regardless" -- the control arm was the right thing to include (this IS exactly what it was designed to catch), and it caught a real confound. A cleaner test would need to hold Wan-side improvement CONSTANT (e.g. freeze the Wan LoRA/cross-attention entirely in both arms, only vary whether the predictor trains) or run joint substantially past where its own oracle/Stage-C gap stabilizes rather than at a fixed 3000-step budget.
Caveats: single run per arm; only 3000 steps (short); FMO/TbD has 4 scenes (low power, several scene-p values n.s.); this doesn't yet compare against the deployed Stage C's own earlier-established gap (~0.03-0.06 corr) since that was measured on a different clip set (28 FMO/TbD clips at 512, not this 68-clip mixed set).

## 2026-09-29 -- CLARIFICATION: naiveburst30k also beats the DEPLOYED 69000-step oracle, not just the weak ref30k stand-in
The 2026-09-29 P2b entry above compared naiveburst30k only against ref30k (also 30000 steps, undertrained). A stronger, more relevant comparison point already exists: P2g's `baseline_oracle` is checkpoint-69000 (the actual deployed oracle used everywhere else in this project) run in oracle mode on the IDENTICAL 68-clip set (same dataset config, epoch_size=68, same FMO/TbD+Adobe240+GoPro sources) -- confirmed by diffing the two eval configs' dataset blocks (byte-identical).
| arm | steps | corr (68 clips, ALL) | PSNR |
|---|---|---|---|
| ref30k | 30000, oracle tokens, from scratch | 0.375 | 25.82 |
| naiveburst30k | 30000, no V-JEPA, raw blur direct | 0.633 | 28.37 |
| baseline_oracle (checkpoint-69000, deployed) | 69000, oracle tokens | **0.455** | **26.51** |
naiveburst30k (0.633) beats even the fully-deployed 69000-step oracle (0.455), not just the undertrained ref30k -- a stronger and more surprising result than the original entry stated. Same caveats apply (this is a raw-blur-conditioning architecture with a much more direct/dense conditioning signal than V-JEPA's 12-tubelet cross-attention summary; not yet paired/scene-clustered against baseline_oracle with a real significance test, since these numbers come from two separately-run evals, not one scorer call against both).
naiveburst is currently continuing to 69000 steps (matching the oracle's own budget) specifically to test whether this holds or shrinks/reverses once naiveburst is trained as long as the oracle -- result pending.

## AspectDiverse_OMoBlur (checkpoint-79000, +10k steps on Adobe/GoPro/FMO/OMoBlur aspect-diverse mix): helps everywhere EXCEPT the dataset it was meant to help (2026-09-30)

Two paired, scene-clustered comparisons against checkpoint-69000 (the deployed oracle), both scored
with `analysis/score_p2.py` (paired per-clip vs the earlier arm, scene-level cluster stats):

**(a) Regression check on the ORIGINAL common set (Adobe240/GoPro/FMO-TbD, 68 clips, 23 scenes) --
`baseline_oracle` vs `aspectdiverse79k_commonset`:**

| dataset | corr (baseline -> aspectdiverse79k) | d | scene p | PSNR (baseline -> aspectdiverse79k) | d | scene p |
|---|---|---|---|---|---|---|
| FMO/TbD | 0.434 -> 0.502 | +0.068 | 0.124 | 33.10 -> 33.09 | -0.01 | 0.958 |
| Adobe240 | 0.490 -> 0.542 | +0.052 | 0.115 | 22.71 -> 23.39 | +0.68 | 0.091 |
| GoPro | 0.450 -> 0.551 | **+0.101** | **0.004** | 21.07 -> 22.25 | **+1.18** | **0.012** |
| ALL | 0.455 -> 0.528 | **+0.073** | **<0.001** | 26.51 -> 27.05 | +0.54 | 0.004 |

**No regression anywhere -- a clear, significant improvement overall (56/68 clips improved), strongest
on GoPro.** energy ratios ~1, zero reversed-direction clips in either arm -- a clean result. This is
plausibly just "10,000 more LoRA steps helps in general," not necessarily anything specific to
aspect-ratio diversity or OMoBlur -- no ablation isolates that yet.

**(b) The actual motivating test -- OMoBlur's own wide-aspect-ratio (5.33:1) test clips, 144-clip
manifest (`findings/omoblur_test_subset/manifest.json`), zero-shot `baseline69k` (ckpt-69000, never
trained on OMoBlur) vs post-training `aspectdiverse79k` (ckpt-79000, trained on it):**

Only 96/144 clips matched across both arms/seeds (48 dropped by the scorer's pairing step -- not yet
diagnosed; possibly the `w0.00`/`w0.50`/`w1.00` decimal in OMoBlur clip stems confusing stem-matching
regexes built for underscore-only names elsewhere in this codebase. Numbers below are the 96 that did
pair; all fell into the scorer's "Adobe240" bucket since `dataset_of()` doesn't recognize OMoBlur stems
-- cosmetic only, doesn't affect the paired comparison itself, just the printed per-dataset label).

| arm | corr | PSNR | energy | reversed-direction clips |
|---|---|---|---|---|
| baseline69k (zero-shot) | 0.211 | 31.36 | 1.76 | 28/96 |
| aspectdiverse79k (trained on OMoBlur) | 0.149 | 29.70 | 1.62 | **60/96** |

**Training ON OMoBlur made OMoBlur's own test clips WORSE, not better**: corr -0.062 (p<0.001), PSNR
-1.67dB. And more than half the clips (60/96) now show reversed-direction motion vs GT, up from 28/96
before training -- this isn't just "less improvement," the energy/direction diagnostic flags a real
motion-quality regression specifically on the wide-aspect data, not merely noise.

**Honest read**: this is a genuinely surprising, not-yet-explained result, and it directly contradicts
the motivating hypothesis ("train on OMoBlur's wide aspect ratio -> OMoBlur gets better"). Two
different, not mutually exclusive possibilities, neither checked yet: (1) something about the
`patch_aspect_ratios` MixedDataset crop-diversity training regime interacts badly with OMoBlur's own
5.33:1 native aspect specifically (the model may be seeing OMoBlur at many DIFFERENT cropped aspect
ratios during training, never the actual full 5.33:1 test-time shape, since `crop_aspect_min/max`
[0.4, 2.5] caps well below 5.33); (2) 10,000 steps at a 0.2 source-weight (1/5 of the mix) may simply be
too few/dilute exposures to this specific data to net out positive, while the OTHER four sources
(weighted 0.8 combined) dominate the update and pull performance toward what helps THEM (matches the
clean win on Adobe/GoPro/FMO in (a) above). Not yet resolved -- needs either a dedicated OMoBlur-only
continuation or a training-data audit of what aspect ratios OMoBlur samples actually got cropped to.

Eval dirs: `OMOBLUREVAL_baseline69k_2026_09_30_13_40_55`, `OMOBLUREVAL_aspectdiverse79k_2026_09_30_14_23_25`,
`OMOBLUREVAL_aspectdiverse79k_commonset_2026_09_29_22_20_06` (all under experiments/infer/).
**Caveat added 2026-10-09:** OMoBlur val reuses the train split's camera setups in 16/16 groups (`Blur2Vid_test_taxonomy_proposal.md` sec. 5), so for aspectdiverse79k these are seen backgrounds with new events, not held-out scenes -- the regression is on seen setups, which makes it stranger, not explained.

## 2026-09-30 -- CAVEAT on P2b (naive burst): NOT a resolution-matched comparison, treat "beats V-JEPA" as unproven
User objection (correct, verified in config/code): the naive-burst branch (`burst_blur_conditioning`,
`_burst_blur_condition_tokens` -> VAE-encode -> `WanBlurCondEmbed`) consumes the raw blur burst at the full
Wan crop (`crop_size`/`out_size` 512), whereas V-JEPA only ever sees frames resized to `vjepa_image_size: 384`
(24x24 token grid, 12 tubelets). So naiveburst30k got ~1.8x the linear resolution (~3x the pixels), i.e. more
raw information, than the arm it "beat". Together with the earlier noted confound (a whole VAE-encoded video vs
a compressed tubelet summary), the +0.258 corr / +2.55 dB over ref30k cannot be attributed to the conditioning
mechanism. Note also ref30k uses ORACLE tokens from sharp GT frames while naiveburst only sees the blur.
Do NOT cite "naive burst beats the V-JEPA oracle" without this caveat. The in-flight `P2_combined_vjepa_burst`
run inherits the same 512 burst input unless its config downsamples it -- check before interpreting it.
Fair rerun needed: feed the burst blurs through the same resize-to-384 (same preprocessing as V-JEPA, then up/
down to the VAE grid as needed) and compare at matched budget; ideally also match token count.

## 2026-10-01 -- P2 combined (V-JEPA tokens + burst-blur conditioning) at 30k, and P2G at 82k: scored
Sources: /scratch/gilbreth/wweligam/p2_scores_combined30k.txt, p2_scores_combined_vs_naiveburst.txt, p2g_scores_82k.txt (all `analysis/score_p2.py`, 68 clips / 23 scenes x 2 sampler seeds, paired, scene-clustered p).
**Combined** (`P2_combined_vjepa_burst`: resumed from ref ckpt-8000, +22k steps = 30k; V-JEPA oracle tokens via cross-attention AND the 512px burst-blur tokens at block 0):
| ALL (68) | corr | PSNR |
|---|---|---|
| nodyn30k | 0.249 | 24.54 |
| dino30k | 0.244 | 24.44 |
| ref30k (V-JEPA oracle) | 0.375 | 25.82 |
| naiveburst30k (burst only, 512px) | 0.633 | 28.37 |
| **combined30k** | **0.672** | **28.88** |
combined vs naiveburst (paired, same 512px burst input in both): corr +0.039 (scene p<0.001, 61/68 clips up), PSNR +0.51 dB (p=0.001); per dataset FMO +0.067 (p=0.060, 4 scenes), Adobe +0.019 (p=0.001), GoPro +0.020 (p=0.001).
Reading: with the burst pixels held identical, ADDING V-JEPA tokens gives a small but consistent gain (+0.04 corr), i.e. the V-JEPA tokens are not useless once the confound is controlled -- they add on top of burst conditioning. The big jump over ref30k (+0.30) is the burst pixels (which both combined and naiveburst receive at 512px vs V-JEPA's 384): that part is still NOT resolution-matched, see the 2026-09-30 caveat. Caveats: single run per arm; combined starts from the ref 8k checkpoint, naiveburst from scratch (same 30k total steps, different lineage); V-JEPA tokens are ORACLE tokens (sharp GT), not Stage-B-predicted.
**P2G at 82k** (joint predictor fine-tune vs predictor-frozen control, +13k from ckpt-69000), ALL corr (oracle / Stage C): baseline 0.455 / 0.418 -> joint72k 0.502 / 0.489, control72k 0.491 / 0.472 -> joint82k 0.537 / 0.512, control82k 0.532 / 0.530. The Stage C gap (oracle - stagec): baseline 0.037, joint72k 0.013, control72k 0.019, joint82k 0.025, control82k 0.002. So at 82k the predictor-frozen CONTROL has the smaller gap; more Wan-side steps help both arms (oracle 0.455 -> 0.53) and joint predictor training shows no clear benefit over the control -- confirms the 72k reading (the gap closure is mostly "more Wan steps"). Single run per arm; FMO/TbD 4 scenes.

## 2026-10-01 -- naive burst-blur conditioning continued to 69k steps: 0.680 corr, far above the deployed V-JEPA systems (same 68 clips)
Source: /scratch/gilbreth/wweligam/p2_scores_naiveburst69k.txt (`analysis/score_p2.py`, 68 clips / 23 scenes x 2 seeds; ref = deployed Stage A ckpt-69000 oracle). The 69k continuation of `naiveburst30k` finished training 2026-09-30 but was never frozen/evaluated (the driver script looked for checkpoint-69000, the final save is checkpoint-69001 -- fixed, frozen, evaluated 2026-10-01).
| arm (ALL, 68 clips) | corr | PSNR | d corr vs deployed oracle | scene p |
|---|---|---|---|---|
| deployed oracle (V-JEPA tokens from GT, 69k) | 0.455 | 26.51 | - | - |
| deployed Stage C (predicted tokens, 69k) | 0.418 | 26.16 | -0.038 | 0.001 |
| naive burst 30k (burst pixels only, no V-JEPA) | 0.633 | 28.37 | +0.178 | <0.001 |
| **naive burst 69k** | **0.680** | **28.94** | **+0.225** | <0.001 |
| combined 30k (V-JEPA oracle tokens + burst pixels) | 0.672 | 28.88 | +0.217 | <0.001 |
Per dataset (naive burst 69k): FMO/TbD 0.588 (4 scenes, p=0.031), Adobe240 0.732, GoPro 0.758 (20/20 clips up on both). Reversal 0/68.
Reading: at the DEPLOYED budget, a model that conditions on the burst pixels alone (no oracle, no Stage B, no V-JEPA) is far better than the V-JEPA-token systems on this protocol, and it keeps improving with steps (0.633 -> 0.680). The resolution mismatch (burst pixels enter at 512px, V-JEPA sees 384px) affects the "tokens vs pixels" attribution but NOT this ranking of deployable systems. V-JEPA tokens still add +0.04 on top at the matched 30k budget (combined vs naiveburst, p<0.001); combined at 69k and a deployable combined (Stage-B-predicted tokens) are NOT run.
Also computed 2026-10-01 (S=1 common-set outputs, reversal = reversed-prediction PSNR > forward PSNR on the 9-frame output vs 9-frame GT window): no-dynamics 22/50 reversed, Stage C 14/50, oracle 1/50 -- vs 0/68 reversed for the burst-3 Stage C and burst-3 oracle on the 68-clip set (different clip sets; same rule).

## 2026-10-02 -- S=1 reversal rate, baselines vs ours (common set, same instrument)
Source: `analysis/reversal_common_s1.py`. Each prediction evenly resampled (linspace) to the GT's 7 frames, then reversed = PSNR(reversed pred) > PSNR(forward pred). Tedla outputs 16 frames, Jin 7, ours 9.
| | GoPro (20) | Adobe240 (20) | FMO/TbD (10) | ALL (50) |
|---|---|---|---|---|
| Jin2018 | 10 | 10 | 6 | 26 |
| Tedla | 7 | 7 | 5 | 19 |
| ours nodyn | 8 | 9 | 4 | 21 |
| ours Stage C | 4 | 9 | 1 | 14 |
| ours oracle | 0 | 0 | 1 | 1 |
(control = blur x7 is time-symmetric by construction; its 24/50 is float-noise tie-breaking, not a rate.) Reproduces the 2026-10-01 numbers (nodyn 22, Stage C 14, oracle 1; nodyn differs by 1 due to the resampling). Baselines at S>1 NOT run (needs per-blur passes + stitching). Caveat: first-frame anchoring in Wan biases this metric toward "forward" (see main report); Tedla/Jin have no such anchor.

## 2026-10-04 — Token-space time flip is NOT pixel-space time reversal (V-JEPA tokens, Stage A / oracle probe)
Source: scripts/viz_jepa_token_pca.py (4 common-set Adobe clips, frozen V-JEPA 2.1 ViT-L, 8-frame clips -> 4 tubelets), figures in /scratch/gilbreth/wweligam/viz_jepa_pca{,_resid}/, cosines.json there. Only 4 clips, no inference statistics.
- Flipping the tubelet axis of V-JEPA tokens (the removed `reverse_time` arm) is not the tokens of the reversed video: V-JEPA fuses each frame pair into one tubelet (a flip does not reverse the pair) and the tokens carry absolute position information.
- Raw token cosines: tokens(reversed video) vs tokens(video) 0.86-0.92; vs time-flipped tokens 0.90-0.93; flipped-vs-original floor 0.85-0.89. The reversed-video tokens are closer to the flipped tokens than to the originals in only 2 of 4 clips (margin -0.012 to +0.071; one clip is a tie, one favours the originals), because a static spatial template (letterbox rows/scene layout) dominates the tokens.
- After removing each position's temporal mean (the time-varying part only): cos(tokens(reversed video), flipped tokens) is 0.20-0.24 on the two clips checked, far from the ~1 that equivalence would require. In one clip the two middle tubelets of the reversed-video tokens line up with the flipped tokens but the end tubelets keep the identity of the forward end tubelets (by eye, 1 clip).
- Consequence: only INPUT-space reversal (`input_reverse`, V-JEPA applied to the backwards video) is a valid reversal intervention for these tokens; token-axis flips were dropped, and any earlier conclusion drawn from them should not be cited.

## 2026-10-04 — Stage A (oracle V-JEPA tokens) input-space reversal: the decoded video reverses (S=1, common set, ckpt-69000, 1280x736)
Setup: Wan input = ONE blurred image (S=1); V-JEPA tokens from the GT sub-clip (8 frames, 4 tubelets) either forward (`none`) or with the clip played backwards BEFORE V-JEPA (`input_reverse`); same sampler seed (1) for both arms. Common-set clips: Adobe 20 (8 scenes), FMO/TbD 10, GoPro 20. Run dirs `/scratch/gilbreth/wweligam/experiments/infer/SUITE_RevSpeedS1_oracle_{adobe,fmo_tbd,gopro}_2026_10_04_*`; scoring `analysis/score_rev_similarity.py`; copied with scores/configs/probe/PCA figures to `/depot/chan129/users/harshana/Blur2Vid/stageA_oracle_input_reversal_S1_2026_10_04/`. GT range bug fixed first (shared manifests use [0,1]; Wan expects [-1,1] -> pm1 manifest copies in /scratch/gilbreth/wweligam/common_testset_pm1/).

| set | arm | dir_cos vs GT | clips reversed (PSNR rule) | PSNR |
|---|---|---|---|---|
| Adobe (20) | none | +0.985 | 0/20 | 27.27 |
| Adobe | input_reverse | -0.651 | 19/20 | 23.45 |
| FMO/TbD (10) | none | +0.741 | 1/10 | 34.73 |
| FMO/TbD | input_reverse | -0.561 | 8/10 | 33.35 |
| GoPro (20) | none | +0.774 | 0/20 | 25.09 |
| GoPro | input_reverse | -0.798 | 20/20 | 20.52 |

- Reversing the video before V-JEPA flips the decoded video's direction in 47/50 clips; the output is closer to `none` played backwards (PSNR 29.2 / 36.4 / 25.8 dB) than to `none` (24.6 / 35.1 / 20.8 dB) on all three sets (FMO/TbD margin small, static background dominates).
- Token level (V-JEPA only, mean over clips): cos(tokens(reversed video), tokens(video)) = 0.894 / 0.904 / 0.895; vs time-flipped tokens 0.907 / 0.898 / 0.909; floor (video vs its own flipped tokens) 0.863 / 0.845 / 0.867. The token-level direction difference is small, yet Wan turns it into a reversed video almost every time, i.e. the V-JEPA tokens DO carry usable direction when the input is reversed in pixel space.
- Caveats: oracle (GT-derived) tokens, 1 seed, 50 clips (8 / 10 / 20 scenes), no confidence intervals; `rev` flag compares against the GT direction, PSNR drops under reversal as expected (the GT forward video is the reference). The Stage B predicted-token (deployable) version of this test is NOT done at S=1; the earlier S=3 predicted-token result is confounded by the burst (see above).

## 2026-10-05 — Evidence-base steps 1-3 (direction-agnostic re-score, token-quality response, S=1 vs S=3 confound)
Scores/tables/code: `/depot/chan129/users/harshana/Blur2Vid/evidence_base_steps123_2026_10_05/` (`analysis/score_steps123.py`, `scripts/make_steps123_configs.py`); run dirs `/scratch/gilbreth/wweligam/experiments/infer/{S1_{nodyn,stagec,oracle}_*,S1blend_*,S3confound_control82k_*,S1_512_control82k_*}`. All Stage A = frozen ckpt (69000 for the common-set runs, P2G control82k for the 512px sets). Metrics per clip: PSNR forward, PSNR of the reversed output ("psnr_r"), best-of-both (a DIRECTION ORACLE: optimistic selection, label it as such), reversal rule (psnr_r > psnr_f), motion-map corr, `dir_cos` = cosine of the output's net optical flow (Farneback) with the GT's. Scenes for clustering = manifest sequences (Adobe 8, GoPro 10, FMO/TbD 4 = 22; an earlier pass mapped GoPro clips to clips and was fixed).
**Data fix**: the 2026-09-27 COMMON_*_fmo_tbd runs had washed-out GT (mean 0.985, PSNR ~8.6 dB for every arm; shared manifests are [0,1], Wan decodes [-1,1]); all S=1 runs here use `/scratch/gilbreth/wweligam/common_testset_pm1/` manifests (GT mean 0.29). Old FMO numbers from those runs must not be quoted.

**Step 1 — S=1 common set (50 clips, 22 sequences, 2 seeds averaged)**
| arm | PSNR fwd | best-of-both | % reversed | corr fwd | dir_cos |
|---|---|---|---|---|---|
| no tokens | 25.88 | 26.20 | 41% | 0.282 | +0.24 |
| Stage B (predicted tokens) | 26.71 | 27.34 | 26% | 0.447 | +0.48 |
| oracle tokens | 27.89 | 27.91 | 2% | 0.551 | +0.85 |
Clustered paired differences: oracle-StageC forward +1.18 dB [0.80,1.54], best-of-both +0.57 [0.38,0.78], corr +0.10; StageC-nodyn forward +0.83 [0.40,1.25], best-of-both +1.14 [0.86,1.43], corr +0.165 (Stage B's gain grows once direction is factored out). Per dataset oracle-StageC forward/best: Adobe 1.68/0.62, GoPro 1.32/0.84; FMO/TbD -0.11/-0.08 (n.s., 4 sequences).

**Step 2 — token-quality / direction response (S=1, Stage B tokens B mixed with oracle tokens O of the forward (F) or time-reversed (R) GT clip; 50 clips, 1 seed)**
| arm | PSNR fwd | best | % reversed | dir_cos |
|---|---|---|---|---|
| B (alpha=0) | 26.60 | 27.32 | 28% | +0.45 |
| (1-a)B + a F, a=.25 / .5 / .75 / 1 | 27.22 / 27.58 / 27.77 / 27.89 | 27.42 / 27.60 / 27.79 / 27.91 | 12 / 2 / 2 / 2% | +0.72 / +0.83 / +0.85 / +0.85 |
| (1-a)B + a R, a=.25 / .5 / .75 / 1 | 25.94 / 24.58 / 24.36 / 24.26 | 27.28 / 27.51 / 27.72 / 27.83 | 48 / 88 / 92 / 94% | +0.07 / -0.54 / -0.67 / -0.69 |
| axis_oracle: B with its coordinate on the F<->R axis set to the oracle forward value | 27.75 | 27.76 | 6% | +0.83 |
| oracle_axis_B: oracle tokens with B's axis coordinate | 26.66 | 27.48 | 28% | +0.42 |
(axis coordinate t(X)=<X-(F+R)/2,F-R>/|F-R|^2, per clip, whole token set.)
- Output quality rises smoothly and monotonically with token quality (even 25% oracle: +0.6 dB, reversals 28%->12%); direction flips at roughly half-way mixing toward the reversed oracle.
- A single scalar per clip, the position on the F<->R axis, accounts for most of the Stage-B-to-oracle gap: replacing it in B recovers 1.15 of the 1.29 dB forward gap (75% of the best-of-both gap), and giving the oracle B's coordinate loses it (26.66 dB). NB the axis is defined from oracle F and R, so this is a statement about WHAT information is missing (direction/mode coordinate, partly also magnitude), not a deployable fix. FMO-only (10 clips) showed a much smaller axis effect.

**Step 3 — S=1 vs S=3 confound**
- S=1 at 512px (68-clip P2G set, Stage A control82k, oracle tokens): input_reverse reverses 93% of clips (dir_cos -0.84, PSNR 26.8 fwd vs 30.2 rev); none reverses 1%. The S=1 flip is not a resolution artifact.
- S=3 (same 68 clips, 2x2 of {V-JEPA input reversed} x {blur burst order reversed}; oracle tokens):
| arm | PSNR fwd | PSNR rev | % reversed (PSNR rule) | corr fwd/rev | dir_cos |
|---|---|---|---|---|---|
| none | 27.09 | 23.25 | 0% | 0.545/0.214 | +0.81 |
| tokens reversed only | 23.32 | 22.29 | 7% | 0.188/0.118 | -0.81 |
| blur order reversed only | 22.64 | 23.78 | 91% | 0.115/0.188 | +0.84 |
| both reversed | 23.35 | 26.87 | 97% | 0.218/0.488 | -0.87 |
  Reading: at S=3 the V-JEPA tokens set the DIRECTION OF MOTION (net flow flips with the tokens: -0.81) and the blur burst order sets the timeline/appearance order; only reversing both gives a clean reversed video (97%, PSNR 26.9 against reversed GT, comparable to the forward arm's 27.1). The earlier S=3 conclusion "reversal does not flip the output" (0% reversed by the PSNR/corr rules) was a metric artifact for token-only reversal: the flow metric shows the motion does reverse.
- Caveats: oracle tokens, 1 seed for steps 2-3; S=3 and 512px sets pool 4 sources (68 clips); dir_cos is a global net-flow cosine (mostly camera motion on Adobe/GoPro); best-of-both is a direction oracle.

## 2026-10-05 — Stroke-controlled Stage B (user-drawn motion strokes -> V-JEPA dynamics tokens), GoPro only: setup
Goal: a single blur is time-symmetric, so Stage B cannot know which way things move (S=1: 26% reversed, dir_cos +0.48 vs oracle 2% / +0.85). Let a USER supply direction: the predictor input is the repeated blur PLUS a sparse, hand-drawable stroke map (lines/arcs with a direction on a person, a car, the background), and it predicts the dynamics tokens matching that control.
- Code: `models/reFlux/stroke_control.py` (RAFT-large keyframe flow -> forward-backward-checked point tracks -> k-means motion groups on (start pos, displacement) -> each sampled group's MEAN trajectory, "user-ified": drawn length 12-45% of the image INDEPENDENT of the true displacement (at 240 fps that is only 1-7%; tying them would leak speed, which must come from N), straight or quadratic-fit arc, +-10 deg, hand wobble, placement jitter; rendered on V-JEPA's 384 canvas through the blur's own letterbox as (dx,dy)*coverage + coverage; `StrokeControlEncoder` = small CNN + per-tubelet tau-signed zero-init heads added at patch_embed AND at block 7's output). `models/reFlux/BlurJepaStrokeControl_model.py` (subclass of BlurJepaDynamics_model). `dataset/gopro_sequence_dataset.py` span mode (`span_len`, raw 42-frame contiguous spans). Configs `checkpoints/MotionBlur2Video/BlurJepaStroke_stageB_gopro{,_SMOKE}.yml`; eval configs from `scripts/make_stroke_configs.py`; scorer `analysis/score_stroke_tokens.py`; tests/visuals `playground/stroke_control/`.
- Training sample: per step (same on all ranks) S~{1,2,3} (.5/.25/.25), N~U{7..14}; S CONTIGUOUS N-frame exposures, blur = plain mean, each blur repeated N times (single_clip); target = ONE frozen V-JEPA forward over the same S*N real frames (odd S*N: both clips repeat their last frame). hflip p=.5 and time reversal p=.5 applied to frames AND strokes as exact transforms (strokes extracted once from the un-augmented clip); 15% rows control-free (S=1: min over forward / TRULY reversed target, not the token-axis flip). Loss = final-layer MSE + 0.5 x layer-normed MSE at block outputs 8 and 12: arXiv 2602.07050 (user-referenced; V-JEPA 2 interpretability) finds motion DIRECTION becomes decodable only around 1/3 depth and fades toward the output, matching our own real-video probe (vy R^2 L8 0.56 vs L24 0.28) -- the final tokens Stage B regresses are where direction is weakest. Warm start: deployed Stage B LoRA (EVAL_FROZEN_StageB_singleclip/checkpoint-20000); control heads zero-init (step 0 == Stage B exactly). S capped at 3 (Stage B went to 5) to keep S*N <= 42 frames.
- Verified before training (`playground/stroke_control/test_stroke_control.py`, 29/29 pass on k050): render(hflip) == hflip(render) with dx negated and render(reverse) negates tangents (<0.05% boundary pixels differ); blur(reversed clip) == reversed burst order; composites leave the blur unchanged; synthetic square moving right over a background moving left gives a rightward and a leftward group, flipped on the reversed clip; zero-init encoder and set_control(None) are bit-exact no-ops through the PeftModel at final and L8/L12; captured block outputs are PRE-norm (final tokens == norms_block[-1](block 23)); odd-N padding aligns predictor/target tubelets. Stroke panels on real GoPro spans (`/scratch/gilbreth/wweligam/stroke_control_viz/`) checked by eye against RAFT flow: directions match; most strokes land on camera/background motion (GoPro is ego-motion dominated). First version drew tiny curly strokes (wobble ~ image size on a 1-7% path) -> drawn length made independent and wobble relative to it.
- Smoke (worst case S=3/N=14 every step): ~4 s/it, 31 GB GPU; save/load roundtrip OK, StrokeControlEncoder_1 in every checkpoint. Full run (mixed S/N) ~1.2-1.8 s/it, 35 GB GPU, ~29 GB anon RAM of the 64G cgroup (+ reclaimable page cache; 0 OOM kills).
- Config trap hit: a FRESH run must use `resume_from_checkpoint: null`; `latest` with `resume_from_path: null` makes base_model `os.listdir(None)` = the cwd, which crashes on the repo's `checkpoints/` dir.
- Pre-registered criteria. Token level (GoPro test spans, scene-clustered): S=1 global +-c dir_acc reaches the v1 global cue's (~0.97 at its training setting); composite `both` (tokens reversed INSIDE one region and forward OUTSIDE) beats the cue's best single direction; shuffle penalty > 0. Wan S=1 E2E (frozen Stage A ckpt-69000, GoPro common set, hires, 2 seeds): stroke_rev reverses >= 75% of clips with dir_cos < 0, stroke_fwd at least as good as plain Stage B on reversal rate / dir_cos with no PSNR loss. Known risks stated up front: (1) every training stroke agrees with ONE global time direction (composites are eval-only, user decision), so the model may learn "read any stroke, apply globally" -- exactly what the composite metric detects; (2) speed is only indirectly trained (a blur is never seen at two N); (3) the E2E input pairing (11-frame blur at R=8) is out of the new model's training distribution but is plain Stage B's own pairing -- the token EVAL scores that pairing separately.
- Training launched 2026-10-05 ~17:37 on gilbreth-k050 (`scripts/run_stroke_stageB_k050.sh`, log `/scratch/gilbreth/wweligam/logs_stroke_stageB_train.log`), run dir `/scratch/gilbreth/wweligam/experiments/Motion/BlurJepaStroke_stageB_gopro_2026_10_05_17_3*`.
- Added after launch (user-referenced "What, Where, and How" V-JEPA 2 trajectory paper, per the user's summary: spatially pooled per-tubelet features trace smooth curved paths; motion decodable at intermediate layers; tangent/spline steering beats linear steering; and from 2602.07050: direction is a high-dimensional population code, not one latent axis). EVAL now also scores the TRAJECTORY readout: cosine between tangents of the pooled per-tubelet paths of prediction vs forward / reversed target (`traj_dir_acc`, `traj_cos`, at final/L8/L12, and region-pooled inside/outside for composites, `traj_both`) -- static scene content cancels, unlike token MSE, which the static spatial template dominates. Optional training term `tangent_loss_weight` (MSE of per-token temporal differences of the layer-normed L8/L12 tokens), default 0 = the running arm; ready as arm B if arm A's direction readout is weak. The control heads already inject a full 1024-d per-token, per-tubelet bias at two depths (a learned multi-dimensional intervention, not a single steering direction).
- Arm A in-training val (18 GoPro test spans, 11 scenes, S=1, deterministic strokes; +c vs -c on the SAME blur): direction accuracy 0.50 with margin ~0.001 through step 2000 (heads growing but no effect), then at step 3000 final-layer dir_acc 0.86 (N=8) / 0.78 (N=12), margin 0.18, shuffle penalty +0.05 (control is used), L12 0.75, L8 only 0.56-0.58. Per the agreed rule (swap to arm B only if the margin was still ~0 at 3k) arm A continues; arm B config (`BlurJepaStroke_stageB_gopro_armB.yml`: mid 10, tangent 10, terms ~1.5/1.0/0.9) is ready but not launched (cannot co-locate on the 64G k050 job). Note arm A's mid-layer term is ~4% of its loss, so its weak L8 readout is expected.
- INTERIM Wan S=1 E2E (arm A ckpt-4000 of 20000, frozen; GoPro common set 20 clips / 10 scenes, frozen Stage A ckpt-69000, 1280x736, seed 1 only; scores `/scratch/gilbreth/wweligam/stroke_e2e_interim4k_scores.txt`, `analysis/score_stroke_e2e.py`): stroke_rev reverses 80% of clips, dir_cos -0.67 (plain Stage B 20%, +0.60; rev-fwd paired dir_cos -1.13, p=0.001, 8/10 scenes) -> the pre-registered "control flips the video" bar (>=75%, dir_cos<0) is MET already. stroke_fwd ~ plain Stage B (20% reversed, dir_cos +0.49 vs +0.60, PSNR 23.74 vs 23.75, all n.s.) -> forward strokes do NOT yet improve on Stage B (bar not met). The v1 global cue (finished, 8k steps, evaluated at its own training pairing blur11/R8) is stronger forward: cue_gt 5% reversed, dir_cos +0.86, PSNR +0.69 dB over Stage B (p=0.015); cue_gt_flip 75% reversed. stroke_none drifted (30% reversed, dir_cos +0.33). Interim only: 1 seed, early checkpoint, blur11/R8 pairing out of the stroke model's training distribution; re-run at the final checkpoint with seeds 1,2.
- PREVIEW token EVAL, arm A ckpt-14000 (frozen), 512 centre crop, 67 GoPro test spans / 11 scenes, scene-clustered (`/scratch/gilbreth/wweligam/experiments/infer/BlurJepaStroke_EVAL_crop512_ck14k_2026_10_05_23_32_12/summary.txt`). Baselines on the same samples/strokes: plain Stage B and the v1 global cue (+-d from GT flow).
  * Global +-c at S=1 (same blur): dir_acc 0.95-0.97 (final), 0.95-0.98 (L8/L12) vs cue 0.99-1.00 (paired -0.00..-0.03, n.s.); trajectory readout dir_acc 0.85-0.93 vs cue 0.80-0.90 (ours >= cue except N=8, the cue's training N). Plain Stage B 0.50 by construction.
  * Token quality: controlled MSE beats plain Stage B's best-of-both-orders at every S=1 cell (-0.07 at N=8 to -0.22 at N=7/14; 10-11/11 scenes) and the cue's controlled MSE at N=7/10/14 (0.73 vs 0.87, 0.75 vs 0.81, 0.90 vs 1.04), tie at N=8 (0.662 vs 0.658). S=3: 0.65-0.87 vs 0.98-1.34 for both baselines -- CAVEAT: the baselines were trained on per-segment targets from overlapping 11-frame windows, this eval uses contiguous exposures and one whole-clip target, so S=3 favours ours by construction.
  * Control is read as direction, not drawing: hflip equivariance gap ~0.00, stroke length x0.5/x2 changes MSE by <0.006; shuffled strokes cost +0.26-0.29 at S=1 (~0 at S=3, where the burst order already sets direction).
  * Wan input pairing (11-frame blur at R=8, Wan's oracle frames 1..8): ours ctrl_mse 0.741 / dir_acc 0.96 vs cue 0.728 / 0.98 vs Stage B 0.937 / 0.50 -> at the token level the stroke model matches the cue on the pairing the cue was trained on. Without a control ours is worse than Stage B there (none_fwd 0.96 vs 0.86): the LoRA drifted away from the uncontrolled task.
  * PER-REGION (composites; one motion group played backwards, blur identical): with ONE stroke, reversing it reverses everything (inside 0.98-1.00 reversed, outside forward 0.03-0.10) -- with no stroke elsewhere the model applies it globally. With >=2 strokes (one reversed, the rest forward) it no longer behaves globally: inside reversed 0.44-0.64, outside forward 0.58-0.68, both satisfied 0.11-0.33 vs the cue's 0.00-0.03 (paired +0.11..+0.33, 3-8 scenes up, 0 down). So a per-region effect EMERGES although training never had conflicting strokes, but it is mostly a compromise; composite training (the user's 'eval + light training' option) is the obvious next lever.
  * Speed via N: NOT read. 14-frame blur at R=7 is no closer to the 2x-speed real frames than to 7 consecutive frames (final 0.61 of samples, MSE diff -0.005 +-0.036; L8 0.18). As flagged at design time: under the chosen N semantics a blur is never seen at two N.
- Arm A training finished 2026-10-06 01:15 (20000 steps, exit 0). In-training val plateaued from ~step 6000: step 20000 S=1 dir_acc 0.94 (N=8) / 0.97 (N=12), margin 0.60-0.62, L8 0.92-0.97, shuffle penalty +0.27-0.28, controlled MSE 0.658 / 0.802 vs uncontrolled best-of-both 0.669 / 0.822. Frozen at `/scratch/gilbreth/wweligam/experiments/Motion/EVAL_FROZEN_StrokeB_armA/checkpoint-20000`. Final evals launched 01:20: k050 = Wan S=1 E2E stroke arms (seeds 1,2) then full-frame token EVAL (`scripts/run_stroke_final_k050.sh`); k047 = Wan E2E v1-cue arms (seeds 1,2), 512-crop token EVAL, direction probe (`scripts/run_stroke_final_k047.sh`); logs `/scratch/gilbreth/wweligam/logs_final_*.log`.

### 2026-10-06 — Stroke-controlled Stage B: FINAL results (arm A, frozen ckpt-20000)
Depot copy (moved by the 2026-10-06 Blur2Vid reorg; see its README.md / MOVES.tsv): E2E `Blur2Vid/stroke_e2e_gopro_S1/` (`stroke_final` = arm A, `cue_*`), token EVALs `Blur2Vid/stroke_token_eval/`, probes `Blur2Vid/stroke_probe/`, frozen ckpts `Blur2Vid/weights/stroke_control_frozen/`.

**Wan S=1 end-to-end** (frozen Stage A ckpt-69000, GoPro common set 20 clips / 10 sequences, 1280x736, seeds 1+2 averaged per clip; references = step-1 runs, same seeds; `/scratch/gilbreth/wweligam/stroke_e2e_final_scores.txt`):
| arm | % reversed | dir_cos | PSNR fwd | best-of-both |
|---|---|---|---|---|
| no dynamics | 40% | +0.13 | 22.75 | 23.12 |
| plain Stage B | 20% | +0.55 | 23.77 | 24.25 |
| oracle tokens | 0% | +0.80 | 25.02 | 25.02 |
| strokes forward | 2% | +0.77 | 24.60 | 24.60 |
| strokes reversed | 95% | -0.67 | 20.79 | 24.29 |
| no strokes (same model) | 22% | +0.41 | 23.76 | 24.49 |
| strokes from another clip | 28% | +0.38 | 23.53 | 24.38 |
| v1 global cue +d | 2% | +0.88 | 24.53 | 24.55 |
| v1 global cue -d | 72% | -0.31 | 21.42 | 24.21 |
- Pre-registered E2E criteria: BOTH MET. (1) reversed strokes reverse 95% (>=75%) with dir_cos -0.67 (<0); vs forward strokes: reversal +0.96 [0.86,1.00], t=31.6, 10/10 sequences; dir_cos -1.36, p=0.006. (2) forward strokes vs plain Stage B: reversal 20% -> 2% (p=0.053, 0/10 sequences worse), dir_cos +0.13 (p=0.17, CI [+0.02,+0.35]), PSNR +0.73 dB (p=0.017, 10/10), best-of-both +0.38 (p=0.010, 10/10). Vs the oracle: dir_cos -0.07 (n.s.), PSNR -0.34 dB (p=0.03) -> ~2/3 of the Stage B->oracle PSNR gap closed by a correct hint.
- CONFOUND + within-model measure: arm A was fine-tuned on GoPro ONLY, plain Stage B and the v1 cue on the Adobe/GoPro/FMO/TbD mix (GoPro weight 0.25), so comparisons to them also reflect domain specialisation (no-stroke best-of-both is already +0.27 over Stage B, p=0.096). The clean effect of the control is WITHIN the stroke model: strokes forward vs no strokes PSNR +0.71 dB [0.29,1.51] (p=0.063; 10/10 sequences up), reversal -17 pts (p=0.14, 0/10 worse), dir_cos +0.30 (p=0.19); strokes reversed vs no strokes: reversal +79 pts (p<0.001), dir_cos -1.06 (p=0.018). Same-size within-model gain for the cue (cue_gt vs cue_none +0.70 dB, p=0.013). Because no-stroke PSNR ~ Stage B (23.76 vs 23.77), the '~2/3 of the gap to the oracle' holds within-model too (0.84 of 1.26 dB).
- Vs the v1 global cue (its own training pairing): forward is a tie (PSNR +0.06, p=0.065; dir_cos -0.11, n.s.; both 2% reversed); reversal is stronger with strokes (95% / -0.67 vs 72% / -0.31; paired dir_cos -0.28, p=0.054, 0/10 sequences favour the cue; reversal +18 pts, p=0.065). No strokes = plain Stage B (PSNR +0.02, n.s.); wrong strokes hurt direction (28% reversed).
- Interim at ckpt-4000 (seed 1): forward strokes were still ~ Stage B, reversed already 80% -> the forward gain comes from later training.
- Caveat (stated up front): strokes are GT-derived, i.e. "with a correct user hint"; the claims are "the control sets the direction of the generated video, both ways" and "a correct hint recovers most of the oracle's gain", not unconditional improvement.

**Token level** (67 GoPro test spans / 11 scenes; same samples/strokes for plain Stage B and the v1 cue; summaries `.../infer/BlurJepaStroke_EVAL_{crop512,fullres}_2026_10_06_*/summary.txt`):
- 512 crop (training geometry): S=1 +-c dir_acc 0.96-0.99 vs cue 0.99 (n.s.); controlled MSE below plain Stage B's best-of-both-orders (-0.05 at N=8, 9/11 sequences; -0.11 at N=10, 11/11); vs cue 0.684 vs 0.658 (N=8, the cue's N) and 0.767 vs 0.813 (N=10); shuffled strokes +0.27; hflip / stroke-length invariance ~0.
- FULL FRAME 1280x736 (the Stage C geometry): control clearly weaker -- dir_acc 0.88-0.89 (cue 0.93, n.s.), margin 0.19-0.21 (vs 0.63 at crop), shuffle +0.08-0.09; controlled MSE vs Stage B best-of-both +0.03 (N=8, worse) / -0.07 (N=10). Likely train/test geometry shift: trained on square 512 crops (stroke 6-15 px thick on V-JEPA's canvas); a letterboxed 1280x736 frame puts the image in a 384x221 strip and the strokes at 3-9 px. Yet the Wan E2E (full frame) works -> the remaining margin suffices for Wan, but training with full-frame / aspect-diverse samples is the obvious fix.
- Direction probe (ridge on oracle token tangents -> per-cell RAFT direction, fit on 300 train spans; `/scratch/gilbreth/wweligam/stroke_probe_armA_20000/direction_probe.json`): oracle reads direction at cos 0.93 (final) / 0.97 (L8) / 0.96 (L12); with strokes, predicted per-cell direction agrees with the stroke at cos 0.93-0.96 (+c) and with the reversed stroke at 0.94-0.96 (-c); +c minus -c = +1.87..+1.92 (max 2). Plain Stage B: |cos| 0.84-0.86 with sign ~random (axis right, direction unknown) -- the strokes supply exactly the missing sign.
- Per-region (composites, FINAL ckpt-20000): with ONE stroke the control acts globally (inside reversed 0.94-1.00, outside forward 0.06-0.20). With >=2 strokes (one reversed, the rest forward) a per-region effect emerges although training never had conflicting strokes: inside-reversed AND outside-forward 0.30-0.34 at 512 crop (cue's best single direction 0.00-0.03; +0.30..+0.34, 5-8 sequences up, 0 down), 0.17-0.38 at full frame (cue 0.00-0.15; +0.10..+0.31, noisier). Still mostly a compromise (inside 0.53-0.82, outside 0.35-0.73).
- Speed via N: not learned (crop 0.63 / full frame 0.24 of samples prefer 2x-speed targets; MSE diff n.s. or wrong sign) -- expected under the chosen N semantics.

**Open / next levers**: (a) full-frame or aspect-diverse training (fixes the 512-crop vs 1280x736 shift); (b) composite training at low probability for per-region control; (c) the k-strided target pairing if speed control is wanted; (d) arm B (mid-layer x10 + tangent loss) not run -- L8 caught up without it, low priority. Ops: a Wan 5B eval was OOM-killed once by the 64G cgroup on k047 while loading (38.5 GB anon) although the identical run passed earlier -- marginal; rerun (seed 2 only; the cue arms do not depend on the stroke checkpoint, so the interim seed-1 cue run was reused). Its log said 'EXIT 0' because MY driver scripts printed `$?` AFTER a `$(date)` substitution, which resets it (fixed: `rc=$?` first; the older `scripts/run_s1_wanlora_k009.sh` has the same pattern). Check OOMs with `dmesg | grep "Memory cgroup out of memory"`. Behaviour change: the Wan S=1 Stage C path now arms the optional v1 cue (`_set_s1_control`); before, only the burst path did, so an older S=1 config with `dynamics_cue_mode != none` now actually applies the cue.

## 2026-10-06 — Stroke-controlled Stage B, PHASE 2 (arm C): continue on GoPro + Adobe240 + X4K + FMO/TbD at varied resolutions, soft-band strokes
Plan: `/home/wweligam/.claude/plans/quirky-hopping-brook.md`. Motivation: arm A (GoPro-only, square 512 crops) loses most of its token-level control at the 1280x736 full-frame (letterboxed, 0.30x) geometry Stage C uses (dir_acc 0.96-0.99 -> 0.88-0.89).
- Data (span mode now in the X4K1000FPSDataset base; GoProSequenceDataset is a generic scene-tree loader with `scenes` / `scene_list_dir` / `span_centers`): GoPro .30, Adobe240 .30 (train_list scenes only), X4K .25 (1000 fps, 65-frame clips), FMO .075 + TbD .075 (train scenes, spans within +-96 frames of the fall). Common-set scenes verified disjoint from training (unit-tested).
- Geometry per sample: 50% whole frame, 50% random crop (aspect 0.5-2, area 20-100%), long side <= 768; batch 1 x grad-accum 2 so shapes can differ per sample. Validation: full frames from all 5 held-out sources.
- Strokes: tracking grid 8 -> 4 (FMO/TbD small objects: spans with >= 1 stroke 0.38 -> 0.75 / 0.88; GoPro unchanged) and a SOFT PERPENDICULAR BAND in the direction channels (user suggestion: Gaussian fall-off sigma = 5% of the shorter side, floor 6 canvas px; line kept in the coverage channel; exact hflip/reverse equivariance re-tested). A thin full-frame stroke now reaches ~37 token cells.
- Warm start: arm A ckpt-20000 weights (LoRA + control encoder), fresh optimizer, global_step 20000 -> 40000. Model hyper-parameters identical to arm A.
- Loader refactor verified byte-identical on the pm1 common-set manifests (GoPro/Adobe/FMO-TbD), an X4K burst sample and the phase-1 GoPro span (`playground/stroke_control/hash_loaders.py`, 11/11 sha256 equal); `playground/stroke_control/test_multires.py` 26/27 (the 1 = a static/saturated random crop with identical consecutive frames, a data property).
- Smoke: ckpt-20000 loads (no load warnings), ~1-1.5 s per optimizer step, peak 59 GB GPU / 40 GB cgroup anon (64G limit). Training launched 2026-10-06 11:46 on k050 (`scripts/run_stroke_multires_k050.sh`, log `logs_stroke_multires_train.log`). Baselines on k047 meanwhile (`scripts/run_stroke_p2_baselines_k047.sh`): arm A + v1 cue Wan S=1 E2E on Adobe / FMO-TbD, arm A GoPro E2E re-run with grid-4 strokes, arm A token EVALs (`scripts/make_stroke_phase2_configs.py`). Every arm is evaluated with the same grid-4 stroke extraction; the renderer follows each arm's training.
- Attribution: arm C changes data + geometry + renderer + steps together -> report as "multi-dataset, multi-geometry continuation with soft-band strokes".
- Ops incidents (2026-10-06 afternoon): (1) the /scratch purge deleted ~6,700 of 34,874 GOPRO_Large_all frames -- the never-read ones (unzip had stamped them 2016; read frames survived on access time). Restored from the intact zip on scratch with `unzip -n -DD` (current timestamps; user's choice; no depot copy). Both arm C and the user's Stage A v2 runs read this data. (2) Arm C was launched onto k050 while the user's `Wan22VJEPADynamics_stageA_v2_base` was already training there (I did not re-check the GPU); they share the A100 at ~2x slower each, user chose to keep sharing (arm C ETA ~03:00). The k047 baseline queue CUDA-OOM'd against the user's `stageA_v2_full` (only my processes died) and the arm-A multi-dataset token EVAL failed on the then-missing GoPro frames; only arm A's GoPro crop512 token EVAL completed. The remaining baselines (arm A multi-dataset token EVAL, arm A / v1-cue Wan E2E on Adobe and FMO/TbD, arm A GoPro E2E with grid-4 strokes) wait for a free GPU.
- Arm C training finished 2026-10-07 04:08 (step 40000, exit 0; 16.4 h wall, shared k050 with the user's Stage A v2_base throughout). Final in-training val (full frames, 8 held-out spans/source, S=1 N=8/12), step 20000 (= arm A) -> 40000: dir_acc GoPro 0.84 -> 0.97, Adobe 0.75 -> 0.97, X4K 0.81 -> 0.92, TbD 0.97 -> 1.00, FMO 0.86 -> 0.93; normalized margin GoPro 0.18 -> 0.28, Adobe 0.11 -> 0.21, X4K 0.20 -> 0.32, TbD 0.17 -> 0.26, FMO 0.05 -> 0.11; most of the gain by step ~23000. Frozen at `experiments/Motion/EVAL_FROZEN_StrokeB_armC/checkpoint-40000`. Final evals queued on k047 (GPU free since the user's v2_full ended): `scripts/run_stroke_p2_final_k047.sh`, logs `logs_p2final_*.log`.

### 2026-10-07 — Phase 2 (arm C, frozen ckpt-40000): FINAL results vs pre-registered criteria
Same grid-4 stroke extraction for both arms; renderer per arm's training (A: line, C: soft band). Token EVALs: `experiments/infer/BlurJepaStroke_P2EVAL_{A,C}_{multi_full,gopro_crop512}_*` (`summary.txt`, per source); probes `/scratch/gilbreth/wweligam/stroke_probe_arm{A_20000,C_40000}_full/`; Wan E2E `experiments/infer/P2E2E_{C_gopro,C_adobe,C_fmo_tbd,A_adobe}_*`, scores `/scratch/gilbreth/wweligam/stroke_p2_e2e_*.txt`.

**Token level, FULL FRAME** (40 held-out spans/source, scene-clustered, paired C-A on identical samples+strokes):
| source | dir_acc S1 N=8 / N=12, A -> C | normalized margin S1N8, A -> C (paired, scenes C:A) | ctrl MSE C vs plain Stage B best-of-both |
|---|---|---|---|
| GoPro | 0.89 / 0.96 -> **0.98 / 0.98** | 0.25 -> 0.33 (+0.08 +-0.03, 11:0) | 0.34 vs 0.41 |
| Adobe | 0.82 / 0.93 -> 0.85 / 0.98 | 0.12 -> 0.16 (+0.05 +-0.04, 7:1) | 0.35 vs 0.40 |
| X4K | 0.85 / 0.88 -> 0.92 / 0.95 | 0.20 -> 0.25 (+0.04 +-0.02, 4:0) | 0.61 vs 0.67 |
| FMO | 0.69 / 0.57 -> 0.96 / 0.86 | 0.04 -> 0.11 (2 scenes) | 0.07 vs 0.16 |
| TbD | 0.96 / 0.93 -> 1.00 / 1.00 | 0.18 -> 0.24 (2 scenes) | 0.05 vs 0.12 |
S=3: dir_acc 1.00 for both arms everywhere (burst order already gives direction); normalized margin up 0.04-0.11. The v1 global cue at full frame: GoPro 0.89-0.90, Adobe 0.88-0.92, X4K 0.97-0.98, FMO 0.80-0.81, TbD 0.77-0.81.
- Direction probe (full frame, GoPro test): predicted per-cell direction vs stroke, final layer: A 0.82 / C **0.91** (+c), 0.85 / 0.92 (-c); L8 0.93 / 0.95; oracle tokens 0.88 (final) / 0.96 (L8); plain Stage B |cos| 0.72 with random sign.
- GoPro 512 crop (phase-1 geometry, 67 spans): C >= A in every cell (dir_acc 0.98 vs 0.96-0.97; normalized margin +0.01..+0.03, 10-11/11 scenes) -> no regression.
- Per-region composites (>=2 strokes, one region reversed): "both" (inside reversed AND outside forward) full frame A 0.31 -> C 0.46 (inside 0.70, outside 0.57 -> 0.76); crop 0.25 -> 0.35; v1 cue 0.01-0.02. One stroke still acts globally (both 0.04-0.08).
- Speed via N: still not read (full-frame fast-wins A 0.30 / C 0.43; crop 0.65 / 0.61 with n.s. MSE differences).
- CAVEAT (bug, fixed 2026-10-07): with val batch size 1 the shuffled-stroke arm used the SAME clip's strokes from the previous cell in every cell after the first -> shuffle penalties are only valid for the first cell (S1N8 here; S1N8 in arm C's in-training val; all phase-1 numbers used batch 2 and are valid). Valid S1N8 full-frame shuffle penalties: GoPro A +0.12 / C +0.15, Adobe +0.04 / +0.05, X4K +0.20 / +0.21. `_shuffled` now keys the previous sample's strokes by cell.

**Wan S=1 end-to-end** (frozen Stage A ckpt-69000, common sets, 1280x736, seeds 1+2):
| set | arm | % rev fwd strokes | dir_cos fwd | PSNR fwd | % rev reversed strokes | dir_cos rev | PSNR no strokes | plain Stage B / oracle PSNR |
|---|---|---|---|---|---|---|---|---|
| GoPro (20) | C | 0% | +0.78 | 24.70 | 100% | -0.74 | 22.73 (45% rev) | 23.77 / 25.02 |
| GoPro | A (phase 1, grid 8) | 2% | +0.77 | 24.60 | 95% | -0.67 | 23.76 (22% rev) | |
| Adobe (20) | C | 8% | +0.82 | 26.78 | 82% | -0.50 | 25.14 (48% rev) | 25.59 / 27.27 |
| Adobe | A | 12% | +0.81 | 26.75 | 85% | -0.49 | 25.27 (48% rev) | |
| FMO/TbD (10) | C | 15% | +0.61 | 34.80 | 60% | -0.37 | 34.63 (25% rev) | 34.84 / 35.11 |
| FMO/TbD | A | 15% | +0.66 | 34.76 | 55% | -0.13 | 34.64 (20% rev) | |
- Criteria: (1) full-frame geometry gap closed -- MET (GoPro dir_acc 0.98 >= 0.95; normalized margin above A, 11:0; probe 0.91 >= 0.9). (2) no crop regression -- MET. (3) Wan GoPro: forward PSNR within 0.2 dB of A (+0.10) and reversed >= 90% (100%) -- MET. (4) Adobe: reversed 82% >= 75%, dir_cos -0.50, within-model fwd - none +1.65 dB (p=0.022) -- MET; FMO/TbD: reversed 60% < 75% -- NOT MET (dir_cos -0.37 < 0 and fwd - none +0.13 dB, n.s., 4 scenes / 10 clips; the PSNR reversal rule is weak when the mover is a sliver of the frame).
- Vs plain Stage B, GoPro +0.83 dB (p=0.010, 10/10), Adobe +1.20 dB (p=0.004, 7/8) with correct strokes -- Adobe within 0.48 dB of the oracle.
- FMO/TbD (4 scenes, 10 clips; scored after the k047 allocation's last minutes): arm A reversed strokes 55% / dir_cos -0.13 vs arm C 60% / -0.37 -- the one E2E place arm C's direction gain shows (dir_cos), on far too few clips to test. Neither arm meets the 75% bar there.
- KEY READING: arm C's clear TOKEN-level gains (full frame, new datasets) do NOT show up end-to-end on GoPro or Adobe -- arm A, never trained on Adobe or full frames, gives the same Wan results (26.75 vs 26.78 dB, 85% vs 82% reversed). At this Stage A, the stroke-induced direction signal was already sufficient for Wan with arm A; the token improvements are headroom the current Wan does not use. Also: arm C WITHOUT strokes picks a direction at random (45-48% reversed; best-of-both unchanged), whereas plain Stage B keeps a forward prior (20% on GoPro) -- arm C trained no-control rows with the min-over-orders loss, so its no-stroke mode is direction-agnostic by design.

## 2026-10-07 — SUMMARY: stroke-controlled Stage B (arm A phase 1 + arm C phase 2), tables + headline conclusions
Arms: **arm A** = stroke Stage B trained 20k steps from the deployed plain Stage B, GoPro only, square 512 crops, line strokes (`EVAL_FROZEN_StrokeB_armA/checkpoint-20000`). **arm C** = arm A continued 20k -> 40k on GoPro + Adobe240 + X4K + FMO/TbD, per-sample whole-frame / random-aspect-crop geometry, soft-band strokes, denser stroke tracking (`EVAL_FROZEN_StrokeB_armC/checkpoint-40000`). (arm B = a planned heavier mid-layer/tangent-loss variant, never run.) "Correct strokes" = synthesized from GT flow, i.e. a correct user hint. Details in the two entries above.

**Generated video, Wan S=1 E2E** (frozen Stage A ckpt-69000, 1280x736, seeds 1+2; cell = PSNR fwd / % clips reversed):
| method | GoPro (20 clips) | Adobe (20) | FMO/TbD (10) |
|---|---|---|---|
| no dynamics tokens | 22.75 / 40% | 25.16 / 48% | 33.60 / 30% |
| plain Stage B (no control) | 23.77 / 20% | 25.59 / 42% | 34.84 / 5% |
| v1 global cue, correct direction | 24.53 / 2% | — | — |
| arm A, correct strokes | 24.60 / 2% | 26.75 / 12% | 34.76 / 15% |
| arm C, correct strokes | 24.70 / 0% | 26.78 / 8% | 34.80 / 15% |
| oracle tokens (upper bound) | 25.02 / 0% | 27.27 / 2% | 35.11 / 5% |
| arm A, reversed strokes | 95% reversed | 85% reversed | 55% reversed |
| arm C, reversed strokes | 100% reversed | 82% reversed | 60% reversed |
| v1 cue, reversed direction | 72% reversed | — | — |

**Token level (V-JEPA predictor), full frame unless noted:**
| metric | plain Stage B | v1 cue | arm A | arm C |
|---|---|---|---|---|
| +-stroke direction accuracy, GoPro (N=8 / 12) | 0.50 | 0.89-0.90 | 0.89 / 0.96 | 0.98 / 0.98 |
| same, Adobe | 0.50 | 0.88-0.92 | 0.82 / 0.93 | 0.85 / 0.98 |
| same, X4K | 0.50 | 0.97-0.98 | 0.85 / 0.88 | 0.92 / 0.95 |
| same, FMO / TbD (N=8) | 0.50 | 0.80 / 0.77-0.81 | 0.69 / 0.96 | 0.96 / 1.00 |
| predicted direction follows the stroke (probe cos, GoPro; oracle 0.88) | sign random | — | 0.82 | 0.91 |
| per-region control ("both" sides right, >=2 strokes) | — | 0.02 | 0.31 | 0.46 |
| GoPro 512-crop direction accuracy | 0.50 | ~0.99 | 0.96-0.97 | 0.98 |

**Headline conclusions**
1. A drawn stroke sets the direction of the generated video: flipping the strokes reverses it in 95-100% of GoPro and 82-85% of Adobe clips -- the time-direction ambiguity a single blur cannot resolve.
2. A correct stroke recovers ~70-75% of the plain-Stage-B -> oracle gap: GoPro +0.83 dB (p=0.010, 10/10 sequences), Adobe +1.20 dB (p=0.004); reversals 20% -> 0% on GoPro.
3. Arm C fixed the full-frame weakness at the TOKEN level (GoPro 0.89 -> 0.98, all new datasets up, no crop regression) but NOT in the video: Wan output equals arm A's (24.70 vs 24.60 dB GoPro, 26.78 vs 26.75 Adobe). The current Stage A does not exploit the better tokens; retraining Stage A on them is the lever.
4. Vs the v1 global-direction cue: tie going forward, clearly better reversing (100% vs 72%), and the only one with per-region control.
5. Not working yet: per-region control is partial (one stroke steers the whole scene; 46% full separation with several strokes); small falling objects reverse only 55-60%; speed via N not learned; without strokes arm C chooses direction at random (~45% reversed) instead of Stage B's forward prior.
Caveats: GT-derived strokes; small test sets (20/20/10 clips, 10/8/4 scenes); arm A vs C also differs in training data. Depot (`/depot/chan129/users/harshana/Blur2Vid/`, reorganized 2026-10-06 -- see its README.md / MOVES.tsv): phase 1 in `stroke_e2e_gopro_S1/`, `stroke_token_eval/`, `stroke_probe/`, frozen ckpts in `weights/stroke_control_frozen/`; phase 2 moved into the same layout on 2026-10-08 (`scripts/reorg_stroke_phase2_depot_2026_10_08.py`, logged in MOVES.tsv): `stroke_e2e_{gopro,adobe,fmo_tbd}_S1/armC_final` (+ `armA_final` for adobe/fmo_tbd), `stroke_token_eval/p2_arm{A,C}_{crop512,multi_full}`, `stroke_probe/stroke_probe_arm{A_20000,C_40000}_full`, `weights/stroke_control_frozen/EVAL_FROZEN_StrokeB_armC`, phase-2 code/configs/logs/plan in `stroke_e2e_gopro_S1/*_phase2`.

## 2026-10-08 — Stroke-controlled Stage B, PHASE 3 (arm D): hybrid speed supervision (setup)
The user's intended semantics for N: the N output frames always span the whole blur trail (first frame = exposure start, last = exposure end), so for a fixed blur fewer frames = faster motion. Arms A/C trained with blur = mean of N frames and target = those N frames: every sample satisfies the endpoint rule, but trail length and N always co-vary, so "span the trail" and "move at the scene's per-frame speed" fit the data equally; the speed test (14-frame blur at R=7) found no reliable speed reading (fast-wins 0.30-0.65, n.s.).
- Arm D = arm C ckpt-40000 + 30% SPEED-PAIR steps: S=1, one blur of an odd E-frame exposure (E in 13..27), supervised at N=E (target: all E frames) AND N=(E+1)/2 (target: every other frame f0, f2, ..., f_{E-1}); both targets share the exposure's first and last frame, so only the endpoint rule fits both. Each branch is backed up separately (memory of one step), weight 1/2. Other 70%: arm C's regular steps unchanged. Hyper-parameters, data mix, soft-band strokes identical to arm C. +12k steps (40000 -> 52000).
- Speed test redefined with exact endpoints: blur of E = 2R-1 frames at R repeats (R = 7 and 14), prediction vs V-JEPA(f[0::2]) (full span, 2x speed) vs R consecutive middle frames; plus token-speed ratio (per-tubelet token change of the R-repeat prediction / E-repeat prediction, next to the same ratio for real frames). In-training val runs the R=7 version (`stroke_val_speed`). Unit test `playground/stroke_control/test_speed_pair.py` (sampler 30% pairs, E odd in 13..27, branch frame counts, reversal commutes with the stride).
- Compute: the user's two allocations were busy, so (user rule) an 8 h normal job (11908642) and a 4 h standby job (11908643) were submitted with a resumable block driver (`scripts/run_stroke_speedhybrid_block.py`: lock so only one trains, resumes the newest > 40000 checkpoint with optimizer state, cancels the sibling when done). Scheduler estimate at submission: start ~2026-10-10 (partition saturated; free-looking GPUs reserved by backfill).
- Arm D training started 2026-10-08 13:18 (job 11908642, gilbreth-i004, A100-80GB): arm C ckpt-40000 loaded cleanly, speed pairs 32% of steps, ~1.1-1.3 s/step (ETA ~17:30). Step-40000 val (= arm C) speed baseline, R=7 (13-frame blur fed as 7 frames), nearer the full-span 2x target: GoPro 0/8, Adobe 3/8, X4K 4/7, TbD 2/8 -> arm C reads N as consecutive frames. DESIGN NOTE: real 2x-speed frames raise V-JEPA's per-tubelet token change by only ~7-23% (ratio_tgt 1.07-1.23), so the token-speed ratio has little range; the stride-vs-consecutive target match is the primary speed readout (`playground/stroke_control/test_speed_eval.py` checks it rewards an ideal speed-reading predictor: fast-wins 1, MSE ~0 at R=7 and 14).
- Eval queue (plan: freeze ckpt-52000; speed-only EVALs for arms D/C/A with R=7,14 on 40 whole-frame spans/source; arm D token EVALs multi_full + gopro_crop512; Wan GoPro S=1 E2E; scoring) is built into `scripts/run_stroke_speedhybrid_block.py` with per-step markers in `/scratch/gilbreth/wweligam/speedhybrid_evals/`; queued as jobs 11908903 (normal 5 h) and 11908904 (standby 4 h), both `--dependency=afterany:11908642`. Pre-registered: (1) speed read -- R=7 and 14 prediction nearer the full-span target in >= 75% of spans with mse(fast)-mse(consec) CI < 0 on GoPro/Adobe/X4K while A/C stay near chance; (2) no direction regression (S=1 dir_acc within 0.02 of arm C per source); (3) Wan GoPro forward-stroke PSNR within 0.2 dB of arm C, reversed strokes >= 90% reversed. Wan Stage A only takes N=8, so speed can only be shown at the token level.
- Arm D training FINISHED 2026-10-08 ~19:15 (checkpoint-52000, full state, in `experiments/Motion/BlurJepaStroke_stageB_speedhybrid_2026_10_08_13_19_00`). In-training val (8 whole-frame spans/source, R=7 speed test) shows NO speed reading after 12k steps (~3.6k speed-pair steps): fraction nearer the full-span 2x target, step 40000 -> 52000: GoPro 0.00 -> 0.12 (mse gap +0.056 -> +0.054), Adobe 0.38 -> 0.25, X4K 0.57 -> 0.57, TbD 0.25 -> 0.38, FMO 0.50 (2 spans). Direction control intact (S1 dir_acc 0.83-1.00). The planned full eval (40 spans/source, R=7/14, arms A/C/D) was NOT run yet: the user cancelled the eval jobs to free the group GPU cap for their v3 training (queue ready: `scripts/sbatch_stroke_speedhybrid_eval_{normal,standby}.sh`). If the full eval confirms, the hybrid pairing at 30% / equal-weight MSE is too weak a speed signal (V-JEPA's tokens barely separate 2x from 1x footage: ratio_tgt 1.07-1.23) -> next lever: a direct margin loss on the pair (pred(N_small) nearer the strided than the consecutive target) and/or a higher pair fraction.

### 2026-10-08 — Phase 3 (arm D, frozen ckpt-52000): FINAL token-level results vs pre-registered criteria
Evals (frozen `experiments/Motion/EVAL_FROZEN_StrokeB_armD/checkpoint-52000`, both EVAL logs load it, no load warnings): speed-only EVALs for arms A (ckpt-20000) / C (40000) / D (52000) on the SAME 178 whole-frame clips (40 spans per source; exact-endpoint speed test: blur of E = 2R-1 frames fed at R repeats, prediction vs V-JEPA(f[0::2]) = full span at 2x speed vs R consecutive middle frames), plus arm D multi_full (whole frames, 5 sources) and gopro_crop512 token EVALs paired against arm C's phase-2 runs. Pairing verified, not assumed: speed runs -- the target-only token-speed ratios (real frames through the frozen target encoder, arm-independent) agree on every row across A/C/D (max |diff| = 0, 169-170 rows), and each speed log loads only its own arm's frozen checkpoint; direction runs -- the frozen baselines evaluated inside both runs (plain Stage B none_best, cue v1 ctrl_mse) agree on every matched row (max |diff| = 0; 969 multi_full rows, 603 crop rows). Paired tool: `analysis/compare_stroke_arms.py` (scene-clustered; `[a:b]` = scenes where first-minus-second is > 0 : < 0). Ran co-located on the user's k019 allocation (train_v3_base), GPU fraction 0.22, OOM-victim priority.

**(1) Speed via N — NOT read (criterion 1 fails everywhere).** Fraction of spans nearer the 2x full-span target (bar: >= 0.75 with mse gap CI < 0 on GoPro/Adobe/X4K):

| source | R | fast-wins A / C / D | D gap mse(2x)-mse(consec) | D-C gap (paired) | D-A gap (paired) |
|---|---|---|---|---|---|
| gopro | 7 | 0.25 / 0.29 / 0.32 | +0.023 ±0.022 | +0.004 ±0.011 [6:5] | -0.024 ±0.007 [0:11] |
| gopro | 14 | 0.24 / 0.27 / 0.34 | +0.013 ±0.026 | -0.011 ±0.008 [3:8] | -0.025 ±0.009 [0:11] |
| adobe | 7 | 0.21 / 0.38 / 0.27 | +0.026 ±0.035 | +0.011 ±0.009 [7:1] | -0.014 ±0.008 [0:8] |
| adobe | 14 | 0.18 / 0.20 / 0.21 | +0.050 ±0.049 | -0.000 ±0.006 [5:3] | -0.009 ±0.009 [1:7] |
| x4k | 7 | 0.49 / 0.44 / 0.41 | +0.012 ±0.147 | -0.005 ±0.033 [2:2] | +0.001 ±0.101 [2:2] |
| x4k | 14 | 0.33 / 0.41 / 0.45 | -0.010 ±0.136 | -0.021 ±0.082 [2:2] | -0.012 ±0.080 [2:2] |
| tbd | 7 | 0.10 / 0.05 / 0.16 | +0.004 ±0.022 | -0.002 ±0.001 [0:2] | -0.003 ±0.022 [0:2] |
| tbd | 14 | 0.02 / 0.13 / 0.41 | +0.003 ±0.034 | -0.005 ±0.016 [0:2] | -0.007 ±0.021 [0:2] |
| fmo | 7 | 0.38 / 0.55 / 0.53 | +0.000 ±0.046 | -0.000 ±0.022 [1:1] | -0.005 ±0.002 [0:2] |
| fmo | 14 | 0.29 / 0.24 / 0.27 | +0.004 ±0.014 | -0.000 ±0.014 [1:1] | +0.000 ±0.044 [1:1] |

All three arms still read a repeat count of R as "R consecutive frames" (gap > 0 on GoPro/Adobe). The one real signal: on GoPro (and Adobe R7) arm D's gap moves toward the stride target in EVERY scene vs arm A (-0.024/-0.025, 0:11), but only partly vs arm C (R14 -0.011, 3:8; R7 +0.004) and the sign never flips -- nudged toward the endpoint rule, nowhere near reading it. With V-JEPA's own 2x-vs-1x token separation this small (ratio_tgt 1.07-1.23), 30% equal-weight MSE pairs are too weak a signal; next lever (not run): a direct margin loss on the pair (pred(N_small) nearer the strided than the consecutive target) and/or a higher pair fraction. Aside: on GoPro 512 crops arm D gives fast-wins 0.61 (R7) / 0.59 (R14), gap -0.004 ±0.032 / -0.013 ±0.040 -- UNINTERPRETABLE without a baseline (arm C's crop512 eval predates the exact-endpoint test; same key names, different definition -- the compare tool refuses to pair them); crop geometry (smaller displacements) may simply bring the two targets closer. Still below the 0.75 bar.

**(2) Direction — NOT met: small but consistent regression vs arm C.** dir_acc mostly equal (S3 cells all 1.000; crop512 S1 0.970-0.977 vs 0.977), but on whole frames several S=1 cells drop by more than the 0.02 bar (point estimates; CIs include 0): GoPro S1N8 0.947 vs 0.977 (-0.030), X4K S1N8 0.872 vs 0.919 (-0.047), X4K S1N12 0.916 vs 0.946 (-0.030), FMO S1N8 -0.021 (2 scenes). The normalized direction margin -- the clause "not lower (paired)" -- is lower almost everywhere, with a reliable sign: GoPro full S1N8 -0.025 ±0.019 [0:11], S1N12 -0.014 ±0.011 [1:10]; Adobe S1N8 -0.016 ±0.014 [1:7], S1N12 -0.023 ±0.012 [0:8]; GoPro crop512 S1N7/N8/N10/N14 -0.018/-0.015/-0.014/-0.004 (0:11, 0:11, 1:10, 3:8); only TbD is higher (+0.018/+0.032, 2 scenes). Meanwhile crop512 ctrl_mse is slightly LOWER for D (-0.016/-0.023/-0.014 at N8/N10/N14, 0:11 / 0:11 / 1:10): arm D fits the targets marginally better and separates the two directions marginally less. Per-region composite 'both': mixed, every CI spans 0 (crop512 N7 +0.09, N8 +0.07, N10 -0.10, N14 -0.01; whole-frame GoPro +0.05, Adobe -0.07/-0.14, X4K -0.13/-0.18). CONFOUND: arm D = arm C + 12k steps; there is no step-matched C' (arm C continued 12k steps without speed pairs), so the margin loss cannot be attributed to the speed pairs rather than the extra training. Shuffle penalty is NOT comparable D vs C (arm C's phase-2 runs carry the batch-1 shuffle bug; only S1N7 on crop512 is valid there: +0.024 ±0.065, n.s.) -- excluded from the comparison by default.

**(3) Wan GoPro E2E — NOT run.** The Wan S=1 step (~80 min, ~47 GB GPU) was skipped to fit next to the user's v3_base training. Given (1) fails and (2) regresses, arm D is not a candidate to replace arm C, so this is low value; config ready (`checkpoints/MotionBlur2Video/Wan22VJEPADynamics_P3E2E_D_gopro.yml`; queue step `wan_D_gopro` + `score_e2e` in `scripts/run_stroke_speedhybrid_block.py`, markers in `/scratch/gilbreth/wweligam/speedhybrid_evals/`).

**Conclusion:** hybrid speed pairs (30%, equal-weight MSE, 12k steps) do not teach V-JEPA's stroke Stage B to read N as speed, and the run ends with a small, consistent loss of direction margin (not attributable to the pairs vs the extra steps -- no step-matched control). Arm C stays the deployed stroke predictor. Depot: `Blur2Vid/stroke_token_eval/p3_speed_arm{A,C,D}/`, `p3_armD_multi_full/`, `p3_armD_crop512/` (each with `cmp_*.txt/json`), `weights/stroke_control_frozen/EVAL_FROZEN_StrokeB_armD/`.

## 2026-10-09 — Local (per-region) stroke control, split by stroke count, per dataset (arms A/C, whole frames)
User decision 2026-10-09: drop the speed-via-N goal; focus on LOCAL stroke control. Re-read of the phase-2 multi_full token EVALs (`P2EVAL_{A,C}_multi_full`, final layer, S1N8+S1N12, composite = one motion group's swept region played backwards, its stroke reversed, any other strokes forward; scene-clustered sample means):

| arm | dataset | 1 stroke: both / in-rev / out-fwd (n) | >= 2 strokes: both / in-rev / out-fwd (n) |
|---|---|---|---|
| A | GoPro | 0.05 / 1.00 / 0.05 (37) | 0.36 / 0.82 / 0.54 (43) |
| A | Adobe | 0.04 / 0.95 / 0.09 (43) | 0.23 / 0.62 / 0.55 (28) |
| A | X4K | 0.27 / 0.98 / 0.30 (25) | 0.32 / 0.60 / 0.72 (26) |
| C | GoPro | 0.02 / 1.00 / 0.02 (37) | 0.33 / 0.62 / 0.71 (43) |
| C | Adobe | 0.14 / 0.98 / 0.16 (43) | 0.59 / 0.81 / 0.79 (28) |
| C | X4K | 0.00 / 1.00 / 0.00 (25) | 0.52 / 0.69 / 0.83 (26) |
FMO/TbD: no composite passes the region filter (2-60% of the frame, separable) -- untested there.
- One stroke acts as a GLOBAL time switch on every dataset (the reversed region reverses, the rest follows it). Arguably expected: an unstroked region has no direction information, but it shows the stroke is not local.
- With >= 2 strokes there is partial local control; more data (arm C) helped on Adobe/X4K (0.23 -> 0.59, 0.32 -> 0.52) but not GoPro (0.36 -> 0.33). Conflicting strokes are resolved by compromise (in-rev 0.62-0.81).
- The failure pattern is the same on every dataset -> it is the training objective, not dataset coverage: time reversal is always applied to the whole clip, so every training sample's strokes agree with ONE global time direction, and "read any stroke, flip everything" is a complete solution. No natural dataset contains one object running backwards relative to the rest; only region composites (`stroke_control.composite_region_reverse`, blur-preserving by construction; also for soft/feathered masks, since any per-pixel blend of a clip and its time reversal has the same temporal mean) supply that signal.
- Generalization to general (real-camera) blur datasets is UNTESTED: every eval so far uses blur = plain sRGB mean of 240 fps frames (noise-free, no clipping/compression, exposure 7-14 frames). Pairs-only real-blur sets cannot provide training targets (targets = V-JEPA tokens of the sharp sequence; strokes = RAFT tracks of it); they can only be scored GT-free (stroke compliance of the generated video's flow, reblur consistency).

## 2026-10-09 — How the stroke condition reaches V-JEPA and Wan (code audit, for local control)
- V-JEPA side (`stroke_control.StrokeControlEncoder`, used by arms A/C/D AND BlurJepaStrokeLocal_model localft/v2): rendered map [3,384,384] -> 5-conv CNN (4x stride 2 = 16 px = one V-JEPA patch; receptive field ~63 px ~ 4 cells) -> per-cell features [576, 256] on the 24x24 token grid, NOT pooled -> per (tubelet, cell) MLP head on [f, tau_t * f] -> ADDITIVE bias on that cell's token, at two depths (patch_embed output; block-7 output). No cross-attention. The map is spatially distinguishable at injection; empty cells get one constant learned bias (CNN of zeros), i.e. "a control is present" but no direction. Nothing restricts propagation afterwards: V-JEPA's 24 blocks of full spatio-temporal self-attention can (and, per the composite evals, do) broadcast one stroke's direction to every token.
- v1 cue (`motion_cue.MotionCueEncoder`): spatial GATE m on the 24x24 grid but ONE direction d per sample -> "where", never "which way per region".
- Wan side (deployed Stage A ckpt-69000: `dynamics_injection: cross_attention`, `dynamics_connector_type: pooled`): under cross_attention the pooled path is NOT used (dynamics_full is only built for value_residual); `_build_dynamics_kv` passes the RAW T x 576 V-JEPA grid (connector applied pointwise) as KV, with learned factorized positions (pos_t[tubelet] + pos_s[cell]) added to K and V. BUT the queries (Wan latent tokens) carry no explicit position in this cross-attention (Wan's RoPE lives only inside its own self-attention), so a Wan token can only find "its" V-JEPA cell by content/implicit position. Whether Stage A uses the dynamics locally is untested; Stage A was trained on oracle tokens of real clips whose motion is mostly global, and earlier analysis found one scalar (the F<->R axis) explains most of the oracle gap -- end-to-end local control may be capped by Stage A regardless of Stage B. Upper-bound test needed: oracle tokens of region-composite clips -> Stage A -> does only the region reverse?

## 2026-10-09 — New data source: FlickrLayerStrokeDataset (natural-image layered motion, exact counterfactual variants)
User request: a separate dataset built on the repo's Flickr2K synthetic motion-blur generator, for local stroke control. `dataset/flickr_layer_stroke_dataset.py`:
- Assets: Flickr2K (`/depot/.../hybrid_Flickr2k_gt_v2`, 2650 photos; train = first 2520, val = last 130) as backgrounds, ForegroundDataset cut-outs (189 train / 4 val) as movers. Camera paths: port of the repo's Boracchi-Foi shake model (`psf/motionblur/trajectory.py`, same kernel-fraction sampling, explicit Generator so a sample is a pure function of (seed, index); directness >= 0.6) or a smooth pan (p 0.4), plus a small roll (<= 1.5 deg) so camera flow varies over the frame. Speeds at 384 px: camera 1.5-7 px/frame, movers 3-9 px/frame (the user's StrokeLocalSynth range).
- Renders the SHARP SEQUENCE (T=8 frames, blur = mean of 29 sub-frames, the Stage-B convention) instead of one kernel-blurred image; strokes come from analytic point tracks through `stroke_control.group_stroke`, i.e. the SAME drawing model as the RAFT strokes on GoPro/Adobe/X4K (length independent of speed, rotation/wobble/shift/thickness jitter).
- Kinds: camera (shake/pan, 1-2 background strokes), objects (static camera, 2 movers, mixed sign pairs; an unstroked mover keeps one random sign in both variants; optional wrong-place stroke on the static background), camera_object (camera + mover, both stroked, forward vs fully reversed only -- over a MOVING background the occlusion order enters the blur, so per-layer reversal is a different blur, not a counterfactual), sharp (random strokes, nothing may move).
- Output contract = StrokeLocalSynthDataset's (blur, clips [V=2,T], strokes [V,K=3,M=16], cell_w, kind, follow_w=1) -> drop-in for `BlurJepaStrokeLocal_model` (`datasets.synth.type: FlickrLayerStrokeDataset`); extras: signs [V,3] per layer, stroke_layer, swept masks [2,H,W] for per-region scoring. Not collatable in one batch with StrokeLocalSynthDataset (M 16 vs 5).
- Verified (`python dataset/flickr_layer_stroke_dataset.py --selftest`): every variant's own playback reproduces the shared blur (max |diff| <= 6.6e-7, all kinds) and its clip; every stroke points along its layer's motion at its location (cos >= 0.98); DataLoader batch -> strokes_from_tensors -> render_strokes runs for train and val. ~0.6 s/sample on 1 CPU core at 384. Optional blur degradations (linear-light averaging, noise, JPEG) are off by default.

## 2026-10-09 — Stage A UPPER BOUND for local control: the frozen Stage A renders ONE global direction, even from oracle tokens
Question (user request): can the deployed Stage A (ckpt-69000, cross-attention injection) render per-object motion at all? Test: the user's synthetic two-mover scenes (dataset/synth_twostroke_dataset.py: 2 unseen objects over GoPro-test backgrounds, 20 scenes, same blur/seed/sampler and scorer `analysis/score_synth_twostroke.py` as their SYN2STROKE armA/armC/v2 runs), but the dynamics tokens are the frozen V-JEPA tokens of each variant's TRUE clip (`Wan22VJEPADynamicsSynthOracle_model`; variant clip = forward frames 0..7 with the reversed object's swept region played backwards -- verified against a direct render: <= 2.3e-5 of pixels off by > 2/255). Runs: `experiments/infer/SYN2STROKE_ORACLE_2026_10_09_09_57_19` (pm, mp, pp) and `..._ORACLE_MM_2026_10_09_10_10_28` (mm). Peak GPU 26.2 GB (A100-80GB, ~6 min per 20 scenes x 3 arms with low_mem_eval).

| run (same 20 scenes) | swap effect | flips as commanded | both objects follow, commanded SAME direction | both follow, commanded OPPOSITE directions |
|---|---|---|---|---|
| Stage B arm A + strokes | 0.45 [0.29, 0.63] | 0.80 | 0.72 (n=29) | 0.21 (n=28) |
| Stage B arm C + strokes | 0.49 [0.29, 0.66] | 0.78 | 0.79 | 0.14 |
| Stage B v2 (user's local FT) + strokes | 0.55 [0.38, 0.72] | 0.80 | 0.90 | 0.18 |
| **ORACLE tokens of the true clip** | **0.50 [0.29, 0.73]** | 0.82 | **1.00** (29/29) | **0.18** (5/28; chance for two independent coin flips 0.25) |
| ORACLE, fully reversed clip (mm) | -- | -- | 0.78 (7/9) | 0.10 (1/10) |
"same / opposite" = sign of cos(commanded image-plane direction of A, of B) (|cos| > 0.3), pooled over the arms pm / mp / none (`analysis/score_synth_twostroke_by_agreement.py`, `/scratch/gilbreth/wweligam/stroke_control_viz/syn2stroke_by_agreement.json`).
- Stage A is a GLOBAL-direction renderer: with oracle tokens of the exact clip it gets both objects right whenever they are asked to move the same way on screen (29/29) and almost never when they must move in opposite directions (5/28) -- in every arm, including the plain forward clip (pp: forward agreement 0.93 when the two objects move the same way, 0.43 when they move opposite ways) and the fully reversed one. The per-scene swap effect is bimodal for exactly this reason (0.82-1.00 in the 11 scenes whose objects move opposite ways forward -- there pm/mp ask for ONE common on-screen direction -- and ~0 in the 9 whose objects co-move).
- Stage B is NOT the bottleneck end to end: v2 tracks the oracle scene by scene (swap effect v2 - oracle +0.05 [-0.04, +0.14], 10:10 scenes; v2 is 0.99 local at token level per the user's eval3). No Stage-B locality gain can reach the video through this Stage A.
- Likely mechanism (confirmed by the probe below): in `WanDynamicsCrossAttn` the V-JEPA keys/values carry learned positions (pos_t, pos_s) but the Wan QUERIES carry no explicit position (Wan's RoPE is applied only inside its own self-attention), so a Wan token can select "its" V-JEPA cell only through implicit position in its content -- which it does not learn to do; the readout is frame-global. Training data (real clips, mostly co-moving/camera motion) would not push against it. The user's running Stage A v3_base / v3_full / v2f_notoken configs use the same `dynamics_injection: cross_attention` without query positions.
- Next: (1) mechanism check -- DONE, see the next entry; (2) fix candidates -- see the next entry (zero-init alpha-scaled distance bias first); (3) Stage A training data with independent / opposite motions (synthetic layered clips, e.g. FlickrLayerStrokeDataset objects kind rendered at Wan resolution); re-run this oracle test as the acceptance check.

### 2026-10-09 — Mechanism: Stage A's dynamics cross-attention is a position-blind (global) readout
Probe (user request): `models/MotionBlur2Video/Wan22VJEPADynamicsXattnProbe_model.py` hooks every block's `WanDynamicsCrossAttn` during ordinary sampling of ckpt-69000 (oracle tokens of the forward clip), recomputes softmax(q k^T/sqrt(d)) exactly as the module does, and measures whether a Wan query at image position (x, y) attends to the V-JEPA cell at (x, y) (letterbox-aware geometry; metrics validated on constructed attention, `playground/stroke_control/test_xattn_probe.py`: a perfectly local pattern gives own-position mass 1.00 vs shuffled 0.002-0.026, a global one own = shuffled, TV 0). Configs `Wan22VJEPADynamics_XATTNPROBE_{gopro,syn}.yml`, runner `scripts/run_xattn_probe.sh`, summaries `analysis/summarize_xattn_probe.py` -> `experiments/infer/XATTNPROBE_{syn_2026_10_09_10_56_53,gopro_2026_10_09_11_10_25}/xattn_probe_summary.{txt,json}` (30 blocks x 50 steps per sample; peak GPU 40.9 GB at 1280x736).

| all 30 blocks, all steps (scene-level means) | real GoPro, 12 clips, 1280x736 | synthetic two-mover, 20 scenes, 512 | perfectly local readout |
|---|---|---|---|
| (mass near OWN position - mass near a RANDOM query's position) / uniform | +0.030 +- 0.006 | +0.044 +- 0.007 | ~ +75 |
| attention centroid vs query position, corr x / y | 0.04 / 0.13 | 0.05 / 0.11 | 1 / 1 |
| query dependence (mean TV to the frame-average attention) | 0.13 | 0.12 | ~1 |
| query latent frame vs attended tubelet, corr | 0.14 | 0.00 | n/a (4 Wan latent frames incl. the deblur prefix are not 1:1 with the 4 tubelets) |
| attention mass on letterbox-padding cells | 8.7% | 0 (square) | 0 |
| cross-attention residual / hidden-state norm | 0.053 | 0.039 | -- |
- Every Wan token reads nearly the same broad mixture of V-JEPA tokens, in space AND time: the dynamics enter as a frame-global code, which is exactly what renders one global direction (previous entry). Same on real in-distribution clips and on the synthetic scenes; stable across denoising phases (early -> late own-shuf 0.035 -> 0.024 on GoPro).
- Only blocks 11-13 show any correspondence, mostly VERTICAL: block 13 corr_y 0.73 (GoPro) / 0.64 (syn) but corr_x 0.14; its best head own-shuf 0.50 -- still far from local.
- Cause consistent with the design: the queries carry no EXPLICIT position (keys/values do). The residual stream does carry implicit position (block 13's vertical correspondence; Wan's RoPE self-attention and conv patch embedding put it there), but the cross-attention almost never uses it, and training on mostly co-moving real clips gave no reason to. The information is present, so the fix should be cheap. The user's Stage A v3_base / v3_full / v2f_notoken runs use the same cross-attention.
- Fix candidates: (a) RESUMABLE first choice: a distance bias on the cross-attention logits in shared, letterbox-aware image coordinates, alpha_h * (-|x_q - x_k|^2), with a learnable per-head scale alpha_h initialised to 0 -- an exact no-op on ckpt-69000 with a nonzero gradient from step 1 (NOT a learnable sigma started large: d/dsigma of -d^2/(2 sigma^2) = d^2/sigma^3 ~ 0, it would never move). (b) 2D(+t) RoPE on q and k in shared coordinates changes the trained logits at once -> needs a gate or rotation of extra dims only to resume cleanly. (c) spatially aligned injection (factored connector + value residual) is NOT resumable: loading the pooled-trained JEPAPerceptualEncoder_1 under 'factored' silently leaves the connector random (module-swap trap), and PGA was never exercised under cross_attention -> needs retraining. Whichever: train on data with independent / opposite motions and use `Wan22VJEPADynamics_SYN2STROKE_ORACLE.yml` (both objects follow when commanded opposite directions; oracle currently 5/28) plus this probe as acceptance tests.

## 2026-10-09 — Stage A LOCAL-MOTION fine-tune launched (v3_full + rank-64 LoRA + cross-attention distance bias + opposite-motion data)
User request: fine-tune Stage A for local motion, increase the Wan LoRA rank, start from v2/v3 weights. Choice: v3_full checkpoint-30000 (newest recipe, finished, LoRA r32 on attention + FFN, partial unfreeze, context frames, blur-every-frame).
- **LoRA 32 -> 64** (alpha 64 -> 128, scale 2 unchanged): `scripts/expand_lora_rank.py` writes an exact expansion (old rows/cols bit-identical, new B columns zero, new A rows kaiming) to `experiments/Motion/EVAL_FROZEN_stageA_v3full_r64init/checkpoint-30000` (300 pairs + 74 modules_to_save tensors). Needed because the generic loader's PeftModel path only PRINTS on a size mismatch and would leave the LoRA untrained.
- **Distance bias** (`WanXattnDistanceBias` in wan_pga.py, wired through WanBlockWrapper; `Wan22VJEPADynamicsLocal_model`): logit += -scale * a[block, head] * D, D = squared query-key distance in V-JEPA cell widths on the shared letterbox canvas, a zero-init (exact no-op at step 0), scale 20. Without a bias the cross-attention path is the original fused SDPA (shared adapter files changed backward-compatibly). CPU tests `playground/stroke_control/test_xattn_distance_bias.py`: zero bias == SDPA (1e-7), nearest key = own cell for every query (square and letterboxed), gradient reaches a at zero.
- **Data**: + `FlickrLayerVideoDataset` (Flickr2K + ForegroundDataset, 1-3 movers, ~60% opposite directions, with/without camera motion; exact v3 sample contract, verified against GoProSequenceDataset) at 30% of batches, the v3 real mix scaled to 70%.
- **Run**: `checkpoints/MotionBlur2Video/Wan22VJEPADynamics_stageA_v3_local.yml`, launched 2026-10-09 11:47 on the user's k012 allocation (GPU was idle after their v3 evals), log `/scratch/gilbreth/wweligam/logs_stageA_v3_local.log`, dir `experiments/Motion/Wan22VJEPADynamics_stageA_v3_local_2026_10_09_11_47_38`, steps 30000 -> 40000 (~1.6 s/step, ~4.5 h), ckpt every 1000 (keeps 5). Verified at start: every module loaded (bias = new module at zero), save/load round-trip PASSED (1046 tensors), checkpoint-0 adapter == the expanded v3 weights (42 sampled tensors bit-identical, same 674 keys); 37 GB GPU, cgroup RAM 51/64 GB.
- **Acceptance** (pre-registered): the oracle two-mover test (`scripts/make_local_eval_configs.py` + `scripts/run_local_evals.sh`): both objects follow OPPOSITE on-screen commands clearly above the baseline (69k: 5/28) while same-direction stays high (69k: 29/29), and the xattn probe's own-minus-shuffled position mass rises well above +0.03. Baseline for this architecture = v3_full ckpt-30000 (job 11915141). Checkpoints 35000 and 40000 are frozen to `experiments/Motion/EVAL_FROZEN_stageA_v3local/` and evaluated automatically (`scripts/watch_v3local_evals.sh`).
- **Baseline (job 11915141, `experiments/infer/SYN2STROKE_ORACLE_v3full_2026_10_09_11_58_26`): v3_full checkpoint-30000 is ALSO global-only** -- oracle tokens of the true clip: both objects follow their command 1.00 (29/29) when commanded the same on-screen direction, 0.21 (6/28) when commanded opposite directions (ckpt-69000: 29/29 vs 5/28); swap effect 0.60 [0.42, 0.78], flips as commanded 0.88. LoRA rank 32, partial unfreeze, context frames and the v3 data (incl. OMoBlur object motion) did not change it.
- **PRE-REGISTERED acceptance criteria (written 2026-10-09 ~12:25, before any fine-tuned result exists; final = checkpoint-40000, 35000 = progress check):**
  1. Locality, oracle two-mover test (same 20 scenes): both objects follow OPPOSITE on-screen commands in >= 14/28 (baseline v3_full 6/28, 69k 5/28), with SAME-direction both-follow >= 26/29 (baselines 29/29).
  2. Mechanism, xattn probe on 12 real GoPro clips (all blocks pooled): own-minus-shuffled position mass >= +1.0 of uniform (69k +0.03; v3_full baseline from job 11915141) and centroid corr_x >= 0.3 (69k 0.04).
  3. Real-data no-regression, paired against the user's EVAL_V3_FULL GoPro and Adobe runs (same 20 clips each, same seed; `analysis/score_video_runs_paired.py`): scene-clustered mean PSNR difference >= -0.3 dB on both datasets, forward AND best-of-direction.
  Caveat to state with any pass: the synthetic test scenes are close to the new 30% training data (pasted cut-outs; only the 4 val objects and GoPro-test backgrounds are unseen), so criterion 2 (real GoPro) and 3 carry the generalisation claim. Rank, bias and data changed together: if logged alpha stays ~0 (mean |alpha| < 0.05) while locality improves, the data did it; a data-only arm (xattn_distance_bias.enabled: false) would separate them.

## 2026-10-09 -- Stage A v2 / v3 recipes, six-agent audit, corrected scores

**Scope.** All runs below are Stage A ORACLE models (`dynamics_source: vjepa_gt`, V-JEPA tokens from the GT exposure frames). None says anything about Stage C. Common set = 20 GoPro + 20 Adobe clips (`/scratch/gilbreth/wweligam/common_testset_pm1/manifest_*.json`, 1280x736, 9 GT frames), scored with `analysis/eval_blur2vid.py`; `*` = better of forward/reversed per clip; paired deltas are scene-clustered (11 GoPro / 8 Adobe scenes), single sampler_seed=1. Differences < ~0.5 dB are inside the noise.

**Recipes (all 30k steps, fresh runs, generated by `scripts/make_pool_configs.py --v2`).**
- v2 = pool (Adobe240, GoPro_Large_all, X4K, OMoBlur, FMO, TbD) in patchified mode with per-batch crop side {384,512,640,704}, isotropic zoom 0.75-1.5, hflip, time reversal; LoRA r=32 (alpha 64) on attn1+attn2+FFN; 4-frame copy prefix (legacy).
- v2_full = v2 + logit-normal t shift 5 (+ matching `inference_flow_shift: 5`) + token dropout 0.2/noise 0.3 + blur latent at every frame + unfreeze patch_embedding/proj_out/condition_embedder/norm2 (fp32 at 0.1x LR) + reblur/pixel losses (those fired on ~10% of steps; dropped for v3).
- v3 = REAL context frames instead of the copy prefix: generated video = [1 condition slot + 4 past | 9 exposure | 3 future] = 17 px frames -> 5 latents; datasets load a 19-frame real window (blur = mean of the centre 11); context-only latent weighted 0.5; fp32 LoRA masters (`fp32_trainable`). v3_full = v3 + tshift5 + tokrob + blur-every-frame + unfreeze (no decoded losses). OMoBlur pool is 2109 vs 6327 samples under v3 (19-frame window drops edge centres; accepted).

**Corrected common-set scores (pm1 inputs).**

| | GoPro PSNR* | LPIPS* | EPE* | reblur | FVD | Adobe PSNR* | LPIPS* | EPE* | reblur | FVD |
|---|---|---|---|---|---|---|---|---|---|---|
| v2_base | 25.28 | 0.149 | 0.466 | 4.16 | 199 | 27.88 | 0.129 | 0.827 | 3.47 | 166 |
| v2_full (shift 5) | 25.41 | 0.144 | 0.398 | 4.28 | 171 | 26.98 | 0.127 | 0.918 | 3.60 | 174 |
| v3_base | 24.81 | 0.137 | 0.425 | 5.24 | 172 | 27.39 | 0.120 | 0.848 | 4.13 | 174 |
| v3_full | 24.82 | 0.134 | 0.379 | 4.64 | 170 | 27.32 | 0.118 | 0.828 | 3.60 | 156 |
| old oracle (ckpt-69000) | 25.09 | 0.152 | 0.437 | 4.41 | 191 | 27.28 | 0.139 | 1.077 | 3.41 | 220 |
| Stage C | 24.27 | 0.166 | 0.699 | 4.44 | 280 | 26.66 | 0.155 | 1.260 | 3.46 | 286 |
| Wan-LoRA-only (30k) | 24.28 | 0.155 | 0.544 | 4.47 | 217 | 26.60 | 0.140 | 1.169 | 3.57 | 254 |
| Tedla | 24.50 | 0.132 | 0.785 | 7.40 | 995 | 26.14 | 0.110 | 1.471 | 5.23 | 1265 |

Paired deltas (`/scratch/gilbreth/wweligam/eval_v2_common20/paired_v3.txt`):
- **Real context (v3_base - v2_base):** LPIPS* -0.012 (GoPro p=0.012) / -0.009 (Adobe p=0.14); PSNR* -0.47 (p=0.095) / -0.49 (p=0.037); EPE* -0.04 / +0.02 (n.s.); **reblur +1.07 (p<0.001) / +0.66 (p=0.007)**. Sharper and more video-like, less blur-consistent, ~0.5 dB lower PSNR.
- **Levers on top of context (v3_full - v3_base):** PSNR/LPIPS ~0; EPE -0.045 (GoPro, p=0.14) / -0.04 (n.s.); **reblur -0.59 (p=0.006) / -0.52 (p=0.11)**. The levers mainly restore blur consistency.
- v3_full - v2_full: GoPro PSNR -0.59 (p=0.07), LPIPS* -0.010 (p=0.010), reblur +0.36 (p=0.02); Adobe n.s.
- Temporal alignment checked: every arm scores best at shift 0 vs GT (no off-by-one in the 17->9 crop); v3_full matches gt[k-1] better than gt[k+1] (slightly slow motion).
- v2_base at 30k already matches the 69k old oracle (new recipe, not the levers).
- Inference shift fix (Euler shift 1 -> 5 for the shift-5-trained v2_full) moved scores by ~0.1 dB only.

**INVALID / do not cite** (found by the 2026-10-08 audit agents; details in memory notes):
- First batch of v2 oracle evals (10-07 / early 10-08; dirs renamed `WASHED_EVAL_V2_*`): eval configs copied `findings/common_testset` manifests that now carry `wan_frames_range: "01"` -> washed GT AND washed V-JEPA oracle input. On GoPro the effect on v2_base/v2_full was <=0.06 dB; on Adobe up to 0.3 EPE.
- 2026-09-27 FMO/TbD oracle row (washed; the "FMO unaffected" note was wrong).
- Jin2018 rows everywhere (736->720 crop + stretch caps PSNR ~22.5 dB alone); plain-Wan-I2V row (5 frames, frame 0 = blur); the v1 `score_common_testset.py` table (truncates 9->7 frames, Tedla 16->7, scene ids wrong).
- "Stage C recovers 79-89% of the oracle" at S=1 and "dynamics does all the work" (vs a token-knockout of the token-trained checkpoint, burst regime); FVD at n=8-20 as a discriminator; calling PSNRp "the Blur2Vid protocol" (it picks direction per frame per patch, averages per-patch dB, 3 of 9 frames).
- v2f_notoken (v2 + fp32 + tokens zeroed) diverged after ~9k steps (loss 0.25 -> 1.55, outputs 0.08 darker); dropped; cause unresolved (no-token instability vs fp32+resume).

**Still valid:** COMMON_* GoPro/Adobe runs (oracle/Stage C/no-dyn); clip<->stem alignment; no resume/optimizer/leak bug in the early Stage A chain; oracle beats Stage C and Wan-LoRA-only by ~0.7-1.3 dB; the forward-plan decision rule FAILS (Stage C not clearly > Wan-LoRA-only at S=1; Stage C had 69k vs 30k steps).

**Code fixes in this round (uncommitted at time of writing):** fp32 masters for LoRA + modules_to_save under bf16 autocast; isotropic zoom; reblur `gt_mean`; connector-family load mismatch raises; `inference_flow_shift`; pm1 range check in `feed_data` (eval raises); strict single-run-dir glob in the scorer; `make_v2_eval_configs.py` rewritten (full arm keys, pm1 manifests, `CKPT=best`); v3 context layout (`context_key`, `context_layout`, `ctx_loss_weight`, `blur_window`/`ctx_key` in the datasets); `train.save_best` + `best_metric` (default EPE) writing `best-<step>` + `best.json`.

**Open.** Why real context worsens re-blur consistency; whether any of this transfers to Stage C (none of these oracles has a Stage B on top); EPE-only best-checkpoint selection can prefer low-PSNR checkpoints; validation clips come from the same test splits as the common set (best-ckpt selection is mildly test-selected).

Paths: scores `/scratch/gilbreth/wweligam/eval_v2_common20/eval_v3_all.{md,json}`, `eval_pm1_with_shift5.md`; eval videos `/scratch/gilbreth/wweligam/experiments/infer/EVAL_{V2_BASE,V2_FULL_SHIFT5,V3_BASE,V3_FULL}_*`; runs `/scratch/gilbreth/wweligam/experiments/Motion/Wan22VJEPADynamics_stageA_{v2_base,v2_full,v3_base,v3_full}_2026*` (checkpoint-30000).
- **v3_full probe baseline** (`experiments/infer/XATTNPROBE_{gopro,syn}_v3full_*`, same probe): equally position-blind -- GoPro own-minus-shuffled +0.023 (69k +0.030), centroid corr_x 0.04, query dependence TV 0.11; synthetic +0.039. On GoPro letterboxed clips v3_full puts 35% of its cross-attention on letterbox-PADDING cells (69k: 8.7%). (The GoPro summary was re-run by hand: run_local_evals.sh had been edited while that job was executing it -- bash reads scripts incrementally -- which garbled the summary line; the probe rows were complete, 18000.)
- Progress (14:56): training at 35495/40000; learned alpha mean 0.073, max 1.26 (max = a strongly local head: at 3 cells the logit penalty is ~11 nats). checkpoint-35000 frozen and its eval job (11915714) running.
- **checkpoint-35000 (5k fine-tune steps) vs the pre-registered criteria** (eval job 11915714; `experiments/infer/SYN2STROKE_ORACLE_v3local35000_2026_10_09_14_48_21`, `XATTNPROBE_{gopro,syn}_v3local35000_*`):
  1. Locality -- MET: both objects follow OPPOSITE commands 0.96 (27/28; v3_full 6/28), same direction 1.00 (29/29); swap effect 0.97 [0.92, 1.00] (v3_full 0.60), flips as commanded 1.00, forward-oracle agreement 0.99.
  2. Mechanism on REAL GoPro -- MET: own-minus-shuffled position mass +4.04 +- 0.04 of uniform (v3_full +0.023; bar +1.0), centroid corr_x 0.77 / corr_y 0.78 (bar 0.3), query dependence TV 0.39 (v3_full 0.11); best head per block +10.8. Synthetic: +7.1, corr 0.48. Letterbox-padding attention still 29% on GoPro (v3_full 35%).
  3. Real-data no-regression -- pending (job 11916019, GoPro + Adobe paired vs the user's EVAL_V3_FULL runs).
  Learned alpha at 35000: mean ~0.07, max ~1.3 -- the bias is used, so it is not "the data alone" (a data-only arm would still quantify the data's share). Caveat stands: test scenes resemble the synthetic training data; criterion 2 is the real-data evidence.
- Training stopped at step 36005 on 2026-10-09 15:12 by SIGTERM right after checkpoint-36000 was saved (exit 143), after the user added query position embeddings (`WanXattnQueryPos`, `xattn_query_pos`) and wrote `Wan22VJEPADynamics_stageA_v4_qpos.yml` (warm start from the frozen v3local checkpoint-35000, distance bias + query positions, to 45000). Not restarted. Run dir keeps checkpoints 32000-36000; frozen copy of 35000 in `experiments/Motion/EVAL_FROZEN_stageA_v3local/`.
- **Criterion 3 (real-data no-regression), GoPro -- MET** (job 11916019, `experiments/infer/EVALREAL_gopro_v3local35000_2026_10_09_20_22_56/paired_vs_v3full.txt`): v3local-35000 vs the user's EVAL_V3_FULL_gopro on the same 20 common-set clips, oracle tokens: PSNR 25.03 vs 24.92, +0.09 +- 0.27 dB (7:4 scenes), SSIM +0.006 +- 0.014. Adobe running.
- v4 (`stageA_v4_qpos`, created and launched 14:59-15:01 by ANOTHER Claude session in this project, on k010; v3local-35000 + query/key (x,y,t) position embeddings, to 45000) finished (checkpoint-45000); not evaluated yet. v3_local's own training on k012 was SIGTERMed at 15:12:53, the minute v4 passed its startup check -- most likely by that session; earlier attributed to the user, probably wrongly.

## 2026-10-09 (late) — v4 acceptance + END-TO-END stroke control through the new Stage A (user request)
- **v4** (v3local-35000 + query/key position embeddings, to 45000; run by another session): oracle two-mover swap effect 1.00, both-ok 1.00 (`SYN2STROKE_ORACLE_v4qpos45000_*`); real-GoPro probe own-minus-shuffled +6.03 (v3local-35000 +4.04), corr_x 0.79 / corr_y 0.83; synthetic +10.3. Real-data no-regression for v4 running on k010 (chain `chain_k010_v4.log`).
- **v3local-35000 real-data no-regression, Adobe**: -0.23 +- 0.59 dB vs v3_full (3:5 scenes), SSIM -0.007 -- inside the pre-registered -0.3 dB bar on the point estimate but borderline (GoPro was +0.09).
- **END-TO-END, synthetic two-mover scenes** (user's StrokeLocal v2 Stage B, analytic strokes -> predicted tokens -> Stage A; same 20 scenes/seed/scorer as SYN2STROKE_v2; `experiments/infer/SYN2STROKE_E2E_v3local35000_2026_10_09_23_10_05`; configs `scripts/make_stroke_e2e_v3_configs.py`): swap effect 0.85 [0.74, 0.95] (old Stage A 69k + same Stage B: 0.55 [0.38, 0.72]); both-ok 0.88 (0.62); flips as commanded 0.95. Stroke arms only: OPPOSITE on-screen commands -> both objects follow pm 1.00 / mp 0.78 (old: 0.11 / 0.22); SAME direction pm 0.90 / mp 0.80 (old 1.00 / 1.00). (The pooled by-agreement line also contains the no-stroke arm, which has no command and should not be read as control.) Per-object stroke control now reaches the video for opposite motions; the same-direction rate dropped ~15 points -- watch on the real-data runs.
- **v4 real-data no-regression, GoPro**: +0.30 +- 0.59 dB vs v3_full (7:4 scenes), SSIM +0.008 (`EVALREAL_gopro_v4qpos45000_2026_10_09_23_08_03`). Adobe running.
- **END-TO-END synthetic, v4-45000** (`SYN2STROKE_E2E_v4qpos45000_2026_10_09_23_44_42`): swap effect 0.97 [0.92, 1.00], both-ok 0.97, flips as commanded 1.00; stroke arms OPPOSITE pm 1.00 / mp 0.89, SAME pm 1.00 / mp 1.00 (v3local-35000 lost ~15 points on same-direction; v4 does not). Paired swap effect: v4 - v3local35000 +0.12 [+0.03, +0.24] (15:5 scenes); v3local35000 - old69k +0.30 [+0.09, +0.49] (16:4); v4 - old69k +0.42 [+0.26, +0.58] (20:0). The query/key position embeddings add on top of the distance bias. Same Stage B (StrokeLocal v2) throughout. Caveat as before: synthetic scenes resemble the synthetic Stage A training clips; the real-data stroke runs (GoPro/Adobe, chains on k010/k012) carry the generalisation claim.
- **v4 real-data no-regression, Adobe -- NOT MET**: PSNR -0.50 +- 0.59 dB vs v3_full (2:6 scenes), best-of-direction -0.41; SSIM -0.006; and 2/20 clips come out TIME-REVERSED with oracle forward tokens (v3_full and v3local-35000: 0/20) (`EVALREAL_adobe_v4qpos45000_2026_10_09_23_42_*`). v3local-35000 on Adobe: -0.23 +- 0.59 (3:5). Both local models lose a little on Adobe and gain on GoPro (+0.09 / +0.30); v4 trades more Adobe fidelity (and some direction stability) for its stronger locality.
- **END-TO-END on REAL clips (global stroke control, no regression check)** -- StrokeLocal v2 Stage B, RAFT strokes from the GT clip (grid 8), arms none / fwd / rev, seed 1, 20 common-set clips (`experiments/infer/STROKEE2E_{gopro,adobe}_{v3local35000,v4qpos45000}_2026_10_10_*`, `analysis/score_stroke_e2e.py`; its NODYN/STAGEC/ORACLE rows are the OLD 69k references):

| Stage A | GoPro fwd PSNR / %rev | GoPro rev-arm reversed | GoPro fwd - none | Adobe fwd PSNR / %rev | Adobe rev-arm reversed | Adobe fwd - none |
|---|---|---|---|---|---|---|
| old 69k + arm C Stage B (2026-10-07, seeds 1+2) | 24.70 / 0% | 100% | -- | 26.78 / 8% | 82% | -- |
| v3local-35000 | 24.76 / 5% | 95% | +0.83 dB (p=0.042, 8/10) | **27.33** / 10% | 95% | +1.76 dB (p=0.017, 7/8) |
| v4-45000 | 24.55 / 0% | 90% | +0.76 dB (p=0.042, 7/10) | 26.72 / 5% | 85% | +1.11 dB (p=0.012, 7/8) |
  (Stage B differs from the old row -- StrokeLocal v2 vs arm C -- and the old row pools 2 seeds; read it as context.)
- **Verdict (2026-10-10):** both local Stage As keep global stroke control on real video at least at the old level (reversed strokes reverse 85-95%, correct strokes beat no strokes by +0.8-1.8 dB). v4 is the more LOCAL renderer (synthetic E2E swap 0.97 vs 0.85; oracle 1.00 vs 0.97; probe +6.0 vs +4.0) but the weaker one on real Adobe (oracle -0.50 dB vs v3_full with 2/20 reversed clips; stroke fwd 26.72 vs 27.33). v3local-35000 is the better real-data model. Real-video PER-OBJECT control is still untested (the real runs only test the global direction); next: an E2E region-composite test on real clips (one moving region reversed, blur unchanged -- `stroke_control.composite_region_reverse`) for both Stage As, and, if v4's locality lead holds there, a v4 continuation with a smaller synthetic share to recover Adobe.

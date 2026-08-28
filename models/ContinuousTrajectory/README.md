# Continuous Trajectory Modeling via Implicit Neural Representations

## Theory

Most deep deblurring methods treat the motion-blur kernel as a discrete 2D
matrix (or a dense per-pixel flow field). Physically, motion blur is the
integral of light hitting the sensor along a *continuous* camera trajectory
`theta(t)` in SE(3) during the exposure window:

```
y = (1 / dt) * int_0^dt  W(x, theta(t))  dt
```

where `W` is a warping function and `x` is the (unknown) sharp scene.
Recovering `theta(t)` from a single blurry image is severely ill-posed if
`theta(t)` is allowed to be an arbitrary function of `t` — that's exactly
equivalent to allowing an arbitrary dense blur kernel/flow field, which is
what most existing methods already do (implicitly or explicitly).

This experiment instead represents `theta(t)` with a conditional **SIREN**
(Sitzmann et al., 2020) — a coordinate-MLP with sinusoidal activations that
is continuous and differentiable in `t` by construction, and has far fewer
effective degrees of freedom than a dense kernel. Restricting the search
space to trajectories this network can express, plus an explicit
low-curvature (smoothness) prior on `theta(t)`, is the mechanism meant to
reduce the ill-posedness of the inverse problem relative to free-form
kernel/flow regression.

## Pipeline

`models/archs/TrajectoryINR_arch.py` (`TrajectoryINR_arch`):

1. **Backbone** (`SharpImageEncoderDecoder`, a small U-Net): blurry image
   `y` → candidate sharp image `x_hat` + global latent code `z`.
2. **Trajectory SIREN** (`TrajectorySIREN`), FiLM-conditioned on `z`: maps
   stratified time samples `t ∈ [0, 1]` → `theta(t) = (rotvec(t), trans(t))`,
   an SE(3) pose relative to `x_hat`'s reference frame. Conditioning on `z`
   (rather than fitting one SIREN per image at test time) keeps the model
   amortized and trainable with this repo's standard batched `trainer.py`.
3. **`pose_to_homography`**: converts each `theta(t)` to a per-time
   homography under a single fronto-parallel scene-plane assumption (the
   standard projective motion-blur model, e.g. Whyte et al. 2012,
   generalized here with a translation/parallax term). Setting the
   out-of-plane rotation and z-translation to zero recovers *exactly* the
   discrete in-plane kernel model already used elsewhere in this repo to
   synthesize training data
   (`psf/motionblur/trajectory2psf.py::get_transformation_matrix`) — i.e.
   that model is the `theta_z`-only special case of this one.
4. **`warp_batch`**: forward-warps `x_hat` by each homography (matching
   `cv2.warpPerspective`'s convention, so these homographies are directly
   comparable to the `Ms` produced by
   `psf/motionblur/trajectory2psf.py::trajectory2psfs`) and averages over
   `t` to synthesize a blurred image `y_hat`.

`models/ContinuousTrajectory/TrajectoryINR_model.py` (`TrajectoryINR_model`)
trains this by **analysis-by-synthesis**:

- `reblur` loss: `||y_hat - y||` — always available, self-supervised, the
  core reconstruction-consistency signal.
- `sharp` loss: `||x_hat - gt||` — used when the dataset provides a ground
  truth sharp image (`SyntheticMotionBlurDataset` does).
- `traj_smooth`: penalizes the discrete second derivative (acceleration) of
  `theta(t)` — the explicit smoothness prior described above.

## Key simplifying assumptions (v1 scaffold)

- **Single fronto-parallel plane.** True parallax depends on per-pixel
  depth; this model folds depth into a single scalar (absorbed into the
  translation magnitude), so it cannot yet express depth-dependent blur
  (e.g. a foreground object blurring differently from its background under
  camera translation). Natural extension: predict a depth map and warp with
  a proper `H(t, depth)` per pixel, or fall back to `models/DeepLensModels`/
  `models/DFlatModels` machinery already in this repo for physically
  accurate optics.
- **Backbone is a small plain U-Net**, not NAFNet/Restormer, to keep this
  scaffold self-contained. Swap `SharpImageEncoderDecoder` for one of the
  existing `models/archs/*_arch.py` backbones once the trajectory-INR idea
  is validated.
- **`train.patched` must be `false`.** The warp needs the full image (it
  pulls content from outside any local crop), so this model does not
  support the patch-based training/inference path other models in this repo
  use for large images.
- **No direct trajectory supervision by default.** The starter config
  (`checkpoints/ContinuousTrajectory/TrajectoryINR_base.yml`) does not
  enable `return_meta` on the dataset, so the model never sees the ground
  truth trajectory (`Ms`) used to synthesize the training blur — matching
  the "recover physics from the image alone" framing. Set
  `datasets.*.return_meta: true` (+ `meta_include_kernel: true`) to also
  pull the GT trajectory through for validation-time comparisons/ablations;
  it isn't wired into any loss yet.

## Running

```bash
conda activate /home/wweligam/home_env
python trainer.py -opt checkpoints/ContinuousTrajectory/TrajectoryINR_base.yml -comment my_run
```

Edit `datasets.train.name` / `datasets.train.foreground_dataset.name` (and
the matching `val` block) in the config first if those cluster paths don't
apply to your setup — they're copied from an existing working
`SyntheticMotionBlurDataset` config in this repo
(`checkpoints/ECCV26/motionReblurLatent.yml`) as a starting point.

## Sanity-checked so far

A standalone forward/backward smoke test (arbitrary, non-multiple-of-8 image
size; custom and default time sampling; train/eval mode) confirms: shapes
are correct end to end, no NaNs, and every parameter receives a gradient.
It has **not** yet been run through a full `trainer.py` loop against real
data — do that before trusting any reported PSNR from it.

## Open questions / natural next steps

- Validate the smoothness-prior weight (`train.lambda_smooth`) and pose
  bounds (`network.max_rotation_deg`, `network.max_translation`) against
  the trajectory generator's own parameter ranges in
  `psf/motionblur/trajectory3d.py::create_trajectory`.
- Add a diagnostic that plots the recovered `theta(t)` (translation/rotation
  vs. `t`) to `log_image` during validation — useful for qualitatively
  checking whether the recovered path looks like a plausible camera shake.
- Consider whether `reblur` alone (no `sharp` supervision) is enough to
  train stably — that's the truly self-supervised, blind-deblurring setting
  the theory motivates, and is worth an ablation once the supervised
  version is confirmed to work.

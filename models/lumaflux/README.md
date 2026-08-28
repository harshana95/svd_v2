# LumaFlux (PyTorch reimplementation)

Paper-faithful implementation of **"LumaFlux: Lifting 8-Bit Worlds to HDR Reality
with Physically-Guided Diffusion Transformers"** (Saini et al., arXiv 2604.02787).
Single-image inverse tone mapping: lift 8-bit SDR (BT.709) to 10-bit HDR
(PQ / BT.2020) by adapting a **frozen FLUX.1-dev MM-DiT** with lightweight,
physically/perceptually guided modules. Only the adapters train (~75M params);
the 12B backbone and the VAE stay frozen.

## Method modules (paper Sec. 4)

| Module | File | What it does |
|---|---|---|
| **PGA** – Physically-Guided Adaptation | `lumaflux/pga.py` | Gated low-rank residual on attention `V`, conditioned on luminance/gradient/saturation + per-head spectral gating. Implemented as a drop-in FLUX attention processor. |
| **PCM** – Perceptual Cross-Modulation | `lumaflux/pcm.py` | FiLM on hidden states from frozen **SigLIP2** embeddings (identity-init residual, bottlenecked). |
| **HDR Residual Coupler** | `lumaflux/coupler.py` | Time/layer-gated additive fusion of physical + perceptual tracks into the latent. |
| **RQS decoder** | `lumaflux/rqs.py` | Monotone rational-quadratic spline tone-field on the frozen VAE output (K=8 knots) + chroma refinement. |
| **Ψ(t, ℓ)** modulation | `lumaflux/modulation.py` | Shared timestep/layer conditioner producing all block-wise gains. |
| Features | `lumaflux/features.py` | Physical descriptors (Tphys, global stats g, FFT bands r) + SigLIP encoder/connector. |
| Loss (Eq. 18) | `lumaflux/losses.py` | Linear-domain L1 on luma + RGB (inverse PQ EOTF) + spline-smoothness, optional flow-matching aux. |
| Wiring | `lumaflux/model.py` | Wraps every FLUX block (PCM+Coupler before, PGA inside), rectified-flow training step + Algorithm-1 sampler. |

## Environment (Gilbreth)

Uses the existing conda env `/home/wweligam/home_env` (torch 2.8 + cu128,
diffusers 0.39, transformers 4.57). FLUX.1-dev and `google/siglip2-so400m-patch16-512`
are already cached under `HF_HOME=/scratch/gilbreth/wweligam/.cache/hub/`.
Local snapshot paths are used in `configs/lumaflux.yaml` because compute nodes are
offline (a repo id would trigger a Hub API call for sharded checkpoints).

```bash
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/wweligam/home_env/bin/python
```

## Data format (pre-paired folders)

```
data/sample/sdr/<name>.png   # 8-bit SDR, BT.709 display
data/sample/hdr/<name>.png   # 16-bit HDR, PQ / BT.2020 signal (or .npy; .exr = linear -> PQ)
```
SDR/HDR share file stems and are pixel-aligned. Generate throwaway samples:
`$PY scripts/make_sample_data.py`.

## Quick start

```bash
# 1. Offline module + wiring smoke test (tiny FLUX, CPU, ~30s)
$PY smoke_test.py

# 2. Make sample SDR/HDR pairs
$PY scripts/make_sample_data.py

# 3. Train (single A100; see SLURM script for batch jobs)
$PY train.py --config configs/lumaflux.yaml --grad_accum 16

# 4. Inference / testing
$PY infer.py --config configs/lumaflux.yaml \
    --ckpt /scratch/gilbreth/wweligam/lumaflux_runs/adapters_final.pt \
    --input data/sample/sdr --output preds --save_hevc
```

SLURM batch training on A100-80GB:
```bash
sbatch scripts/train_a100.slurm --grad_accum 16
```

## Validated

End-to-end tested on a Gilbreth **A100-80GB** with the real FLUX.1-dev:
training step (forward+backward+optimizer) and inference both run; 75.5M trainable
/ 12.49B total; **~30 GB peak** at 256–512px (batch 1); outputs are genuine 16-bit
PQ HDR. Gradient checkpointing keeps memory ~constant across resolution.

## Implementation notes / deviations

These are faithful-in-spirit choices where the paper underspecifies, made so the
frozen backbone stays usable and training is stable:

- **Flow convention.** FLUX/SD3 rectified flow: `z=(1-σ)·data+σ·noise`, backbone
  predicts velocity `v=noise-data`. SDR enters only via the adapters, so the
  pretrained prior stays meaningful. The pixel-space Eq. 18 loss is computed on a
  one-step data estimate (`z-σ·v`) decoded through VAE+RQS. An optional small
  flow-matching aux (`fm_weight`) stabilises early training (set 0 for paper-exact).
  `init_from_sdr` offers an SDEdit-style start at inference (Algorithm 1 line 4).
- **Prompt-free.** Empty/zero text tokens by default; `use_real_null: true` uses
  real `""` T5/CLIP embeddings.
- **PCM/Coupler** are identity-initialised residuals with bottleneck MLPs — this
  preserves the frozen prior at step 0 and keeps the adapter budget "minimal"
  (75M vs a naive 2.2B with full `dim×dim` layers).
- **RQS** predicts one monotone tone curve per image from a pooled latent
  (global tone field); chroma refined by a 1×1 conv.
- **Gradient checkpointing** requires the conditioning context to remain populated
  through `backward()` (blocks are recomputed there), so `training_step` does not
  reset it.

## Not included (paper scope beyond this reimpl)

- The dataset *construction* pipeline (TMO/codec degradation, Eq. 19) — this repo
  consumes pre-paired folders by design. `SyntheticPairDataset` exists only for
  offline pipeline tests.
- Eval metrics (PU21 PSNR/SSIM, HDR-VDP3, ΔE_ITP) and multi-GPU training
  (`accelerate launch`) are left as follow-ups.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch-based image/video deblurring and restoration research framework. It supports physics-based optical deblurring (DFlat lens arrays), motion deblurring, diffusion-based super-resolution, and one-step diffusion models. Training uses HuggingFace Accelerate for multi-GPU distribution, OmegaConf for configuration, and Comet ML / W&B for experiment tracking.

## Environment

```bash
conda activate /home/wweligam/home_env
```

Key packages: PyTorch 2.8 + CUDA 12, diffusers 0.35, transformers 4.57, accelerate 1.1, datasets 3.3, omegaconf 2.3, basicsr 1.4.

## Running Training

**Single GPU:**
```bash
python trainer.py -opt checkpoints/<project>/<experiment>.yml
```

**Multi-GPU (Accelerate):**
```bash
accelerate launch --num_processes <N> trainer.py -opt checkpoints/<project>/<experiment>.yml
```

**Flags:**
- `-test`: Sets project/name to `test` for quick validation runs
- `-infer`: Sets `is_train=False` for inference-only mode
- `-comment "label"`: Appends a label to the experiment name

**Testing/Inference a saved model:**
```bash
python test_saved_model.py -opt <config.yml>
```

**Computing metrics on saved outputs:**
```bash
python calculate_metrics.py --pred <dir> --gt <dir> --metrics psnr ssim
# or via shell script:
bash scripts/calculate_metrics.sh
```

## Architecture

The framework uses a **plugin architecture** where models, network architectures, and datasets are auto-discovered by scanning for `*_model.py`, `*_arch.py`, and `*_dataset.py` files. Classes are instantiated by name from the YAML config.

```
trainer.py
  └─ models/__init__.py         # Discovers and instantiates model_type from config
       └─ BaseModel (base_model.py)
            ├─ setup_dataloaders()     # Reads datasets.{train,val} config keys
            ├─ prepare()              # Optimizer, scheduler, accelerator
            ├─ train()                # Training loop with gradient accumulation
            └─ validation()           # Metrics + image logging
                 └─ [Subclass]        # Implements feed_data(), optimize_parameters(), validate_step()

  dataset/__init__.py           # Discovers and instantiates dataset type from config
  models/archs/__init__.py      # Discovers arch by scanning for {type}_arch class
```

### Model Families

| Directory | Models | Description |
|---|---|---|
| `models/` | `Simple_model`, `HF_model`, `LD_model`, `Sparse_model` | Baseline deblurring (single/multi input) |
| `models/Diffusion/` | `Diffusion_TwoInput_model`, `DiffusionLKPN_*` | Diffusion-based restoration |
| `models/OneStepDiffusion/` | `OSEDiff_*`, `PixelSpace_*` | One-step diffusion (SD / SD3 based) |
| `models/DFlatModels/` | `DFlatAllFocus_model`, `DFlatArrayDepth_model` | DFlat optical lens array models |
| `models/DeepLensModels/` | `DeepLensArrayE2E_model` | DeepLens end-to-end models |
| `models/MotionBlur2Video/` | `Blur2Vid_CogVideoX_*`, `Blur2Vid_Wan_*` | Blur-to-video (CogVideoX / Wan) |
| `models/SyntheticDeblurring/` | `SimpleSynthetic_model` | Synthetic blur deblurring |

### Network Architectures (`models/archs/`)

Auto-discovered `*_arch.py` files. Common types: `Restormer`, `NAFNet`, `LKPN`, `PSF`, `STN`, `WienerDeconv`, `DeblurDeform`, `fftformer`, `MSArray`, `Encoder`, `Identity`. Diffusion models use `UNet2DConditionModel`, `AutoencoderKL`, `T2IAdapter` from `diffusers`.

### Dataset Types (`dataset/`)

- `HuggingFaceDataset`: Loads from HF Hub; most commonly used
- `SyntheticMotionBlurDataset`: Generates motion blur on-the-fly from local image folders
- `DeepLensDeblurDataset`: Physics-based PSF simulation using GeoLens
- `QISCMOSDataset`: Quanta Image Sensor / CMOS sensor simulation

## Configuration System

All behavior is defined in YAML files under `checkpoints/{project}/{experiment}.yml`. OmegaConf handles variable interpolation with `${var}` syntax.

**Key top-level fields:**
```yaml
name: <project_name>
model_type: <ClassName>_model      # Must match class in models/**/*_model.py
is_train: true
num_gpu: 8
report_to: comet_ml                # or wandb, tensorboard
mixed_precision: no                # bf16, fp16, or no
allow_tf32: true

network:
  type: <ClassName>                # Must match *_arch.py file
  # ...arch-specific params

datasets:
  train:
    type: <DatasetClass>
    transforms:                    # Applied as pipeline; keys specify which sample fields
      to_tensor: {keys: [gt_key]}
      crop: {keys: [gt_key], h: 512, w: 512}
      resize: {keys: [gt_key], factor: 0.5}
      rotate: {keys: [gt_key], angle: 90}
      apply_model: {keys: [lq_key], model_name: NAFNet_arch, model_path: /path}
  val: ...

train:
  loss: 1*MSE+0.1*Gradient         # Weighted sum of loss terms
  batch_size: 8
  max_train_steps: 20000
  validation_steps: 100
  checkpointing_steps: 500
  checkpoints_total_limit: 2
  patched: false                   # Enable for large images
  patch_size_h: 256
  patch_size_w: 256

val:
  metrics:
    psnr: {crop: 30}
    mse:  {crop: 30}

path:
  root: /scratch/.../experiments
  resume_from_checkpoint: latest   # or null, or path
```

**Experiment outputs** are saved to `{path.root}/{name}/{experiment_key}_{comment}_{timestamp}/`.

### Loss Specification

The `train.loss` string is parsed into a weighted combination:
- `1*MSE` — mean squared error
- `1*L1` — L1 loss
- `0.1*Gradient` — gradient loss
- `0.01*VGG` — perceptual loss
- Multiple terms: `1*MSE+0.1*Gradient+0.01*VGG`

## Adding New Components

**New model:** Create `models/<name>_model.py` with a class named `<Name>_model` extending `BaseModel`. Implement `feed_data()`, `optimize_parameters()`, and `validate_step()`.

**New architecture:** Create `models/archs/<name>_arch.py` with class `<Name>_arch`. Optionally add `<Name>_config` dataclass for typed config.

**New dataset:** Create `dataset/<name>_dataset.py` with class `<Name>Dataset` (PyTorch Dataset subclass).

All three are auto-discovered — no registration step needed.

# Unfreezing the V-JEPA encoder in Stage-A: what actually blocks it

Status: 2026-09-01. Written while verifying the "F1 — unfreeze the encoder inside Stage-A"
item, which the plan estimates at **~15 lines**. The two lines it names are correct, but they
are not sufficient, and the most likely failure mode is silent.

## What the plan says, and what is true

`ContextProjectorAlign_model.py:270` does hard-code `self.vjepa.requires_grad_(False)`
(followed by `.eval()`), and `setup_optimizers` does collect only projector parameters:

```python
trainable_params = [p for p in self.projector.parameters() if p.requires_grad]
if self.target_bridge is not None:
    trainable_params.extend(p for p in self.target_bridge.parameters() if p.requires_grad)
```

Both verified. Flipping those two is genuinely small.

## The blocker: `torch.inference_mode()` on the extraction path

`_extract_vision_features` (`ContextProjectorAlign_model.py:551`) wraps the entire V-JEPA
forward pass:

```python
def _extract_vision_features(self, pre_01):
    with torch.inference_mode():
        vjepa_pixels = vjepa_from_unit_tensor(...)
        return extract_vjepa_tokens_multilayer(self.vjepa, vjepa_pixels, ...)
```

`torch.inference_mode()` is **stricter than `torch.no_grad()`**. Tensors produced inside it are
flagged as inference tensors and cannot participate in autograd *at all*, even once execution
leaves the block — using one in a graph that requires grad raises
`RuntimeError: Inference tensors cannot be saved for backward`.

So after unfreezing, one of two things happens, and the second is worse:

1. The run errors out at the first backward pass, or
2. it appears to train — loss falls, checkpoints are written — while the encoder receives **zero
   gradient** and only the projector actually learns.

Case 2 is the same shape of silent no-op as the PEFT resume bug that made every resumed LoRA run
train nothing. Do not trust a falling loss curve as evidence the encoder moved: assert on
`vjepa.<param>.grad is not None` and on a nonzero weight delta between step 0 and step N.

There are nine `inference_mode` sites in this file; only the vision-feature one at 551 is on the
F1 path, but `_extract_dino_features` (568) and `_extract_clip_features` (583) have the identical
shape if those encoders are ever unfrozen.

## Second risk: dtype and activation memory

`self.vjepa_dtype` defaults to **fp32** — deliberately, per the comment at the load site: the
hub weights use fp32 attention and bf16 inputs cause Q/K/V dtype mismatches. A frozen fp32
ViT-L/384 forward already sits at ~71.7 GB on the 80 GB A100 running Stage-A. Adding backward
activations for a ~300M-parameter encoder on top of that is a real OOM risk, not a rounding
error. Expect to need gradient checkpointing on the encoder, a smaller `vjepa_image_size`, or a
resolved bf16 path — none of which is in the 15-line estimate.

## Related blocker, unrelated cause: the stale HF dataset cache

`Deblur_MIOR/val` now holds **100 pairs** on disk (up from 11), but the cached Arrow build still
records `validation: 11`, because `Deblur_MIOR.py` declares `VERSION = datasets.Version("1.2.1")`
and `datasets` keys its cache on `(name, config, version, hash)`. Analysis scripts read image
directories directly so they see the new split; anything going through `load_dataset` does not.
Bumping the loader `VERSION` forces a rebuild under a new key and leaves the old build intact.

Until that is done, any Stage-B or Stage-A arm validates on 11 images — the sample size the plan
itself says cannot separate anything.

## Suggested order

1. Bump the dataset `VERSION` (seconds; otherwise every later number is measured on 11 images).
2. Remove `inference_mode` from the encoder path *and* add a gradient assertion, before adding
   the config flag.
3. Size the memory with a short smoke run before committing GPU-days.

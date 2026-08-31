# VAEBlurAlign: findings so far

Status: 2026-08-31. Model: `models/reFlux/VAEBlurAlign_model.py`. Dataset:
`dataset/fullres_patchifier_dataset.py`. Configs: `checkpoints/reFlux/VAEBlurAlign*.yml`.

## The idea

Train a LoRA-adapted FLUX.2 Klein VAE encoder, JEPA-style: splice one
rectangular block of the real blurred image into an otherwise-clean GT crop,
and supervise the encoder to make that block's latent tokens (and decoded
pixels) match GT — using the surrounding clean context as the only extra
information available. The question: does having clean context to condition
on actually help recovery, compared to a full-frame blur→GT baseline with no
context at all (the `_fullblur_baseline` arm, same mechanism with
`replace_ratio=1.0`)?

## Timeline of what was tried, and what each thing fixed or didn't

1. **Shared encoder, stop-gradient target** — the same LoRA-adapted encoder
   produced both the trainable branch and the (stop-gradiented) target
   branch. Training plateaued fast and noisily: latent loss dropped from
   ~0.40 to ~0.15-0.20 within ~50-60 steps then never moved again over
   500-6000 steps, r≈-0.07 (essentially no trend). Diagnosis: even with
   `.detach()`, the target's *forward value* still drifted every step because
   it shared the same live LoRA weights as the branch being trained — the
   encoder was chasing a target it was itself moving.
   **Fix**: a second, fully separate, permanently frozen copy of the
   pretrained VAE (`self.vae_gt`) for the target only. Did not fix the
   plateau by itself (see #2).

2. **batch_size=1, single-example gradient steps** — an overfit-one-sample
   sanity check proved gradients/optimizer were fine (`loss_latent`
   0.283→0.132 over 40 steps on a single repeated example), but population
   training over thousands of distinct (crop, block) tasks showed ~zero
   trend. Each real training step was one gradient computed on a brand-new
   task, with no repetition to consolidate on.
   **Fix**: `FullResPatchifierDataset` (an `IterableDataset`) — loads
   `batch_size` full-res images once, draws `patches_per_group` crops from
   them sharing one randomly-drawn shape (so they can actually `torch.stack`
   into a real batch), amortizing disk I/O. Real batch_size=4 ×
   gradient_accumulation_steps=4 = effective batch 16, with real GPU
   parallelism instead of 16 sequential single-image passes. This also fixed
   a live bug: `FullResBlurDataset`'s per-sample random crop aspect ratio
   made `val.batch_size > 1` crash ("stack expects each tensor to be equal
   size") the moment two samples in a batch landed on different shapes.

3. **VGG perceptual loss + full-frame reconstruction-anchor loss** — replaced
   with a plain block-cropped pixel MSE loss, and the reconstruction-anchor
   term (a third encoder pass, GT→encode(trainable)→decode→compare to GT,
   meant to keep the trainable encoder's general fidelity honest) was
   dropped entirely. This *caused* a regression that had actually been
   latent all along: validation showed the decoded output was measurably
   **worse than doing nothing** — MSE(pred, GT) > MSE(mixed, GT) in the block
   region, 11-12/12 sampled validation images. Removing the reconstruction
   anchor removed the only term constraining the encoder's general
   decode-fidelity outside the directly-supervised region.

4. **GAN loss (3 attempts)** — added a PatchGAN discriminator
   (`MultiscaleDiscriminator` from `models/archs/related/LD/`, hinge loss)
   judging the full decoded frame against real GT, hypothesizing checkerboard
   decoder artifacts were the cause of #3's regression.
   - Attempt 1 (`gan_loss_weight=0.02`, matched G/D learning rate): D fully
     collapsed G by the end of 750 steps (`loss_d`→~0.35, `loss_gan`→~2.0+,
     monotonic the whole run). Block-region fidelity got *worse* over
     training (one sample's gap grew from +0.00028 at step 300 to +0.00050 at
     step 750), never beating baseline in 12/12 samples at any checkpoint.
   - Attempt 2 (`gan_loss_weight=0.005`, `discriminator_learning_rate=5e-5`,
     10x below G): same directional collapse, just slower (D loss 0.43-0.60
     vs 0.38-0.43 at step 300; more oscillation near the end, 0.34-0.46
     rather than monotonic to zero) — but the actual outcome metric still
     didn't improve step 300→750 (mean delta +0.00013→+0.00015), still
     12/12 worse than doing nothing.
   - **Conclusion**: dropped the GAN loss (`gan_loss_weight: 0`) — three
     full-length attempts never beat the baseline even once, regardless of
     D/G balance.

5. **Root-cause diagnosis: the "worse than baseline" regression was never a
   training bug.** An overfit-per-sample sweep (fresh LoRA reset, 20
   independently-sampled real images, ≤130 dedicated steps each) found:
   - Only 9/20 samples ever crossed below the do-nothing floor.
   - The 11 that never crossed had floor MSE (blur vs GT, [-1,1] pixel scale)
     mostly 0.00027-00.0019. The 9 that crossed easily (often within 0-10
     steps) had floor MSE mostly 0.0034-0.0093.
   - Measured directly: the **frozen, untrained VAE's own**
     `decode(encode(gt))` round trip costs **0.0005-0.0019 MSE** on a clean
     image with zero adaptation — an architectural noise floor from lossy
     encode/decode alone.
   - **Conclusion**: for any block whose true blur-vs-GT signal is smaller
     than the VAE's own round-trip noise, beating "do nothing" is
     mathematically impossible — not a training failure, a ceiling. A large,
     uncontrolled fraction of every batch (and of the fixed validation panel)
     was an unwinnable comparison by construction. This explains every prior
     "worse than baseline" result and why patched vs. fullblur-baseline never
     separated: real signal was drowned out by architecturally-capped noise.

6. **Block-floor filtering** — `FullResPatchifierDataset._sample_block` now
   rejects/resamples block placements below `min_block_floor_mse` (default
   0.002, with margin over the measured VAE floor), retrying up to
   `max_block_resample_attempts` (default 20) times before falling back to
   the strongest signal found. Only partially effective on its own: a direct
   measurement on the val split still showed 38% of selected blocks below
   threshold, because retrying block *position* within an already
   blur-free *crop* can't manufacture blur that isn't there.

7. **Crop-position biasing** — the deeper fix. Motion blur is usually
   localized (a moving subject against a sharp background), so a uniformly
   random crop position often misses it entirely regardless of how the block
   is placed within that crop. `_load_pair` now precomputes a per-pixel
   blur-energy integral image once per loaded full-res pair;
   `_pick_crop_position` uses it to score every valid crop position in one
   vectorized pass and sample among the top-20 highest-energy positions
   (not a pure argmax, to keep some placement diversity). Verified directly:
   this dropped below-threshold blocks from **38% → 11%**, and roughly
   quadrupled the median block floor MSE (0.0028 → 0.0101).

## Where things stand after crop-position biasing (750-step run, both arms)

- Latent loss shows the strongest trend of the whole investigation:
  r=-0.28 (patched) / -0.22 (fullblur baseline), down ~26%/~14% from start
  (0.49→0.36, 0.43→0.37). Pixel loss is now roughly flat (r=0.05-0.07)
  instead of trending upward like every prior configuration.
- Block-region fidelity: on the fixed 12-sample visualization panel, only
  2/12 beat the baseline — but 7 of those 12 fixed positions are *still*
  below the recoverability threshold even after crop-bias (an artifact of
  always visualizing the same first-N samples, not representative of the
  ~89% pass rate measured directly). Restricting to the 5 panel samples that
  are actually above threshold: **2/5 (40%) show real improvement**, and one
  of them kept improving between step 300 and step 750
  (delta -0.00049 → -0.00076).
- This is the first configuration in the whole investigation to produce any
  winning examples at all, and the winners keep improving with more
  training steps — consistent with the overfit diagnostic's prediction.
- Patched vs. fullblur-baseline are still statistically close throughout
  this run; the original "does clean context help" question remains open.

## Open questions / natural next steps

- Re-run validation metrics restricted to above-floor-threshold samples only
  (or just draw a larger/fresh validation panel now that crop-bias is in
  place) — the fixed 12-sample panel is a poor read given the selection-bias
  problem noted above.
- Run substantially longer (the 750-step budget was sized for fast
  iteration between architecture fixes, not for judging convergence) now
  that the recipe is stable: separate frozen target encoder, real batching
  (batch_size=4 × grad_accum=4), MSE-only losses (no GAN), block-floor
  filtering, crop-position biasing.
- Once/if the model reliably beats baseline on above-threshold examples,
  re-examine whether patched beats fullblur-baseline — that comparison was
  never cleanly resolvable while most of the training signal was
  architecturally-capped noise.
- Considered and declined (for now): feeding the block mask as an explicit
  input channel to the encoder (analogous to how V-JEPA's *predictor* — not
  its context encoder — receives mask tokens + positional embeddings at
  target locations). Would require widening `conv_in` from 3→4 channels,
  a structural change LoRA deltas can't express on their own.

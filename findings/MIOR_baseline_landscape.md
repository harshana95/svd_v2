# MIOR Baseline & Dataset Landscape (prioritized shortlist)

Scope: single A100-80GB budget. Priority given to methods with public GitHub code
AND released pretrained weights, so zero-shot eval or LoRA fine-tuning is cheap.

## Prioritized baseline shortlist

| # | Method | Category | Status | GitHub | Weights |
|---|---|---|---|---|---|
| 1 | NAFNet | Regression (ECCV'22) | Done — ~28.2 dB per ICASSP_plan.md | github.com/megvii-research/NAFNet | Yes (GoPro width32/64) |
| 2 | Restormer | Regression transformer (CVPR'22 Oral) | Queued next (per ICASSP_plan.md) | github.com/swz30/Restormer | Yes (GoPro-trained; zero-shot HIDE/RealBlur numbers published) |
| 3 | FFTformer | Frequency-domain transformer (CVPR'23) | Arch already in `models/archs/related/`; needs a config | github.com/kkkls/FFTformer | Yes (weights checked into repo: GoPro + RealBlur) |
| 4 | OSEDiff | One-step LoRA diffusion (NeurIPS'24) | Not started; recommend LoRA fine-tune only (not full retrain) — fairest comparison since it's also a LoRA-adapted single-step diffusion prior, similar compute profile to our own method | github.com/cswry/OSEDiff | Yes |
| 5 | DPS or BlindDPS (zero-shot, no training) | Measurement-consistency posterior sampling (ICLR'23 / CVPR'23) | Free — pure inference on an existing pretrained diffusion checkpoint, no fine-tuning needed | github.com/DPS2022/diffusion-posterior-sampling, github.com/BlindDPS/blind-dps | Yes (uses public pretrained DDPMs) |

**Why this order:** #1–3 are the standard regression-baseline trio already anchoring the
paper's Experiments section — finishing FFTformer gives three architecturally distinct
non-diffusion points (CNN / windowed-transformer / frequency-transformer) for minimal
extra cost since the arch is already implemented. #4 (OSEDiff) is the one diffusion-prior
baseline worth actually training, because its LoRA-on-frozen-backbone recipe is the
closest apples-to-apples comparison to our own method's training cost — a full DiffBIR or
LucidFlux retrain would not fit a single A100 in reasonable time. #5 is essentially free
(zero-shot, no training) and gives a "naive measurement-consistency without a learned/
aligned encoder" contrast point that directly motivates why context-conditioning
(our method) should beat blind posterior sampling.

**Optional stretch (only if time remains, evaluate off-the-shelf, do not retrain):**
DiffBIR (github.com/XPixelGroup/DiffBIR) and LucidFlux (github.com/W2GenAI-Lab/LucidFlux)
— run their public pretrained checkpoints directly on the Deblur_MIOR val/test split
without fine-tuning, clearly caveated in the paper as "off-the-shelf, not adapted to MIOR."

**Closest novelty precedent (cite, don't necessarily benchmark):** GramSR
(arXiv:2604.25457, github.com/aimagelab/GramSR) and PixRestore (arXiv:2608.16793,
github.com/csslc/PixRestore) both condition a diffusion/flow restorer on frozen DINO
features — the closest published mechanism to ContextFlux. No paper found using
V-JEPA specifically as a frozen restoration-conditioning encoder, which supports the
novelty claim for that encoder choice. Differentiate on: (a) our JEPA-style masked-block
LoRA adaptation of the encoder itself (vs. their untouched off-the-shelf encoders), and
(b) motion deblurring in a FLUX.2-Klein rectified-flow pipeline vs. their generic SR focus.

## Dataset comparison shortlist

| Dataset | Action | Why |
|---|---|---|
| Deblur_MIOR | Keep (primary) — **verify citation** | See flag below |
| GoPro | Keep (already used) | Standard generalization test; every Tier-1/2 baseline above ships GoPro-trained weights directly comparable |
| HIDE | **Add**, zero-shot only, no training | Standard cross-dataset generalization protocol (train on GoPro/MIOR → zero-shot test on HIDE); all shortlisted baselines already report HIDE zero-shot numbers |
| RealBlur-J | **Add**, zero-shot only, no training | Tests generalization to genuinely real (non-frame-averaged) camera blur — addresses the likely reviewer concern that GoPro/MIOR blur is synthetic/averaged rather than true real-world blur; all shortlisted baselines publish RealBlur-J weights |
| REDS, BSD, RealBlur-R, DPDD | Skip | Video-focused (REDS, BSD) or wrong degradation type (DPDD = defocus, not motion blur) — low value for a single-image motion-deblurring paper |

Adding HIDE + RealBlur-J costs nothing beyond download + eval scripting — no new
training runs — since every Tier-1/2 baseline's public weights already report numbers
on both.

## Flagged decisions needing author input before submission

1. **MIORe/Deblur_MIOR citation accuracy.** `Deblur_MIOR`'s pair count (7,849 train
   pairs, 1920x1080) is a near-exact match to the MIORe deblurring-track release
   (Ciubotariu et al., "MIORe & VAR-MIORe," ICCV'25 Oral, arXiv:2509.06803 —
   github.com/george200150/MIORe, data via the AIM'25 Challenge Track 1 Google Drive
   link — 7,860 train / 80 val / 75 test). MIORe is a **multi-task** benchmark
   (deblur + video frame interpolation + optical flow) and its blur is generated by
   **adaptive averaging of real 1000fps footage**, not a kernel-based synthetic blur
   model — the same construction convention as GoPro/HIDE, but worth stating precisely
   rather than calling it "synthetic" outright. `papers/main.tex` already has an
   `\authnote` flagging this as unconfirmed — resolve before submission.
2. **FlowSteer self-citation.** arXiv:2512.08125 ("FlowSteer: Conditioning Flow Field
   for Consistent Image Restoration"), co-authored by Harshana Weligampola and Stanley
   H. Chan, is a DPS-style zero-shot flow-conditioning restoration method evaluated on
   deblurring — this looks like the lab's own prior/concurrent paper. Decide whether to
   treat it as a related-work self-citation or fold it into the zero-shot baseline row
   (#5 above) alongside/instead of DPS/BlindDPS — it costs nothing extra either way
   since it's zero-shot.

## Follow-up: additional candidates found (deblur-specific diffusion, FLUX-as-backbone)

Prompted by "what about DeblurDiff / a Flux baseline?" — found several more, mostly
better task-fits than OSEDiff for the diffusion-prior slot:

| Method | Category | Status | GitHub | Weights |
|---|---|---|---|---|
| **FideDiff** | One-step diffusion, deblur-specific (2025) | Newest in this family (supersedes OSDD); consistency-model reformulation + Kernel ControlNet for blur-kernel estimation | github.com/xyLiu339/FideDiff | Yes — code + pretrained + dataset released |
| **OSDD** | One-step diffusion, deblur-specific (2025) | Predecessor to FideDiff; eVAE + dynamic dual-adapter | github.com/xyLiu339/OSDD | Yes |
| **DeblurDiff** | SD-prior + Latent Kernel Prediction Network (NeurIPS'25) | Same author group as FFTformer (`kkkls`) already in this repo | github.com/kkkls/DeblurDiff | Yes — checkpoints on Google Drive, MIT license |

**Recommendation: swap FideDiff in for OSEDiff (baseline #4 in the shortlist above)**,
or run both if time allows. OSEDiff is a super-resolution method adapted for this
comparison; FideDiff/OSDD are purpose-built one-step diffusion deblurring models with
released weights — same "efficient single-step diffusion" category, better task fit.
DeblurDiff is a good second diffusion-prior point if a second slot is available (also
from a familiar codebase author).

**Cite-only (no code/weights released, not directly benchmarkable):**
- **DeblurFlow** ("Restoration-Aligned Generative Flow Models for Blind Motion
  Deblurring," arXiv:2605.08854, Kim & Shin) — LoRA-adapts a pretrained **SANA-0.6B**
  flow-matching model via a reformulated "r-space" residual flow trajectory; evaluated
  on GoPro/HIDE/RealBlur/RWBI (33.05–33.69 dB). Methodologically the closest published
  analogue to our own "LoRA-adapt a pretrained flow-matching DiT for deblurring"
  approach — cite prominently in related work, but no repo found to benchmark against.
- **FluxIR / FluxGen** ("Acquire and then Adapt: Squeezing out Text-to-Image Model for
  Image Restoration," arXiv:2504.15159) — a lightweight SE-layer adapter that controls
  **FLUX** (the same T2I DiT family we build on) for restoration, broadcasting control
  into all MM-DiT blocks. Confirms "adapt FLUX as a restoration backbone" is an active
  direction, but it's evaluated only on SR benchmarks (DIV2K/RealSR/DrealSR/RealLQ250),
  not motion deblurring, and no code is released — cite as related work only.

**The most relevant "FLUX baseline" is actually already ours to run, not a paper.**
The obvious ablation is: LoRA fine-tune the FLUX.2-Klein transformer directly on
blurry-image conditioning, with no VAE-encoder JEPA-alignment stage (Phase 1 skipped).
This already exists conceptually as the **"No-JEPA control"** entry in the ContextFlux
ablation family — but per [[contextflux_checkpoints_destroyed]], the whole
encoder-comparison + No-JEPA checkpoint family was lost to the scratch purge (only
BlurMap survived, now on depot). **Action: confirm whether No-JEPA control needs to be
rerun before it can be reported as a baseline number.**

A free additional sanity check: **FLUX.1 Kontext [dev]** (open-weight,
github.com/black-forest-labs/flux) supports instruction-based image editing
(e.g. "remove the motion blur") zero-shot, no training — cheap to add as an
"off-the-shelf FLUX, no restoration-specific adaptation at all" floor.

## Sources

- MIORe/VAR-MIORe: arxiv.org/abs/2509.06803, github.com/george200150/MIORe
- Regression baselines: github.com/swz30/Restormer, github.com/megvii-research/NAFNet, github.com/kkkls/FFTformer
- Diffusion-prior: github.com/cswry/OSEDiff, github.com/XPixelGroup/DiffBIR, github.com/W2GenAI-Lab/LucidFlux
- Posterior sampling: github.com/DPS2022/diffusion-posterior-sampling, github.com/BlindDPS/blind-dps, arxiv.org/abs/2503.08136 (FlowDPS), arxiv.org/abs/2512.08125 (FlowSteer)
- Encoder-conditioned precedent: arxiv.org/abs/2604.25457 (GramSR), arxiv.org/abs/2608.16793 (PixRestore), arxiv.org/abs/2312.01677 (DINO-IR)
- Datasets: seungjunnah.github.io/Datasets/gopro.html (GoPro), HIDE (Shen et al., ICCV'19), cg.postech.ac.kr/research/realblur (RealBlur)
- Deblur-specific diffusion (follow-up): github.com/xyLiu339/FideDiff, github.com/xyLiu339/OSDD, github.com/kkkls/DeblurDiff, arxiv.org/abs/2605.08854 (DeblurFlow), arxiv.org/abs/2504.15159 (FluxIR/FluxGen), github.com/black-forest-labs/flux (FLUX.1 Kontext)

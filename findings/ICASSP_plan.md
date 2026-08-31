# ICASSP Paper Plan — JEPA-Based Context-Aware Flux for Image Deblurring

**Drafted independently from the experiment record (configs in `checkpoints/reFlux/`, models in
`models/reFlux/`, analysis in `playground/`) BEFORE reading `papers/main.tex`.**
Target: submission-ready in 2 weeks (by ~Tue Sep 9, 2026; ICASSP hard deadline expected mid-Sept).

---

## 1. The story (as the experiments support it)

**Claim:** Self-supervised V-JEPA features extracted from the *blurred* input are substantially
blur-invariant, and can be aligned into the conditioning space of a frozen text-to-image flow
model (Flux 2 Klein 4B) to act as "semantic context" that replaces the text prompt — improving
diffusion-based motion deblurring without captions and without finetuning the backbone.

Three supporting pillars, each already having experimental evidence or infrastructure:

1. **Why JEPA (analysis):** SVD spectrum + blur-invariance-per-component + GT-vs-blur token
   cosine similarity for V-JEPA vs DINOv2 vs CLIP (`playground/analyze_jepa_svd.py`,
   `compare_jepa_features.py`, outputs in `playground/jepa_svd/`). This becomes Fig. 1 / the
   motivation section — the "usability of JEPA for deblurring" analysis.
2. **How (method):** Two-stage recipe.
   Stage A — `ContextProjectorAlign`: dual-head patch cross-attention projector maps frozen
   V-JEPA 2.1 tokens → (i) Klein text-embedding space (supervised by Qwen3-VL captions through
   the frozen text encoder) and (ii) packed VAE latent space (MSE + perceptual).
   Stage B — `ContextFluxBaseline`: frozen Flux 2 Klein; projected JEPA tokens appended to the
   null-prompt text context (text head) and to the latent token sequence (VAE head). NOTE
   (verified in code 2026-08-26): the projector is the ONLY trainable module — despite the
   config comment, there is no LucidFlux-style condition branch, and with
   `use_condition_image_for_transformer: false` the LQ latents never enter the transformer, so
   projected JEPA tokens are the sole carrier of image content.
3. **That it works (results):** encoder ablation (JEPA vs CLIP vs DINO vs VL-JEPA vs
   no-conditioning LoRA), layer-depth ablation (last 1/2/4/8 blocks), source ablation
   (LQ vs GT-oracle upper bound), projector-head ablation (text/vae/both), on Deblur_MIOR.

**Honest-narrative fallback:** if JEPA does *not* beat CLIP/DINO decisively, the paper can still
stand as a controlled study — "which frozen visual encoder makes the best prompt for a frozen
diffusion restorer, and what does blur-invariance analysis predict about it" — but that framing
decision must be made by end of Week 1 when full-test-set numbers are in.

## 2. Gap analysis — what exists vs what the paper needs

| Paper element | Status | Action needed |
|---|---|---|
| Motivation feature analysis (JEPA/DINO/CLIP under blur) | Scripts + saved SVD arrays exist | Re-run over a *set* of images (not 1), make publication figure |
| Stage-A/B method | Implemented, trained (JEPA, CLIP, DINO, VLJEPA runs exist on scratch) | Freeze best checkpoints; write method section |
| Encoder ablation runs | Runs exist (06/07–06/21) | **Full test-set eval missing** — val was 5 batches, PSNR/MSE only |
| Layer-depth ablation | Only Layers4 run exists; Layers2/8 configs but **no runs found** | Decide: run them (2× ~50k-step A100 jobs) or drop axis to {1,4} |
| GT-source oracle | Config exists, **no run found on scratch** | High value (upper bound); launch immediately if kept |
| LQ-aligned projector (`JEPA_LQ`) | Rerun launched today (08/26) | This closes the GT-align/LQ-inference domain gap — track it; it may become the headline config |
| No-JEPA control | NoJEPA_LoRA runs exist (05/30) | Verify comparable training budget; re-eval on full test set |
| External baselines (NAFNet, Restormer, DiffBIR/LucidFlux-class) | Repo has NAFNet/Restormer training + `calculate_metrics.py` | Train-or-eval 2–3 baselines on Deblur_MIOR test split |
| Perceptual metrics (LPIPS, DISTS, FID, NIQE/MUSIQ) | `calculate_metrics.py` / `analysis/` support them | Standardize one eval script over saved outputs of every method |
| Qualitative figure | Val images saved during training | Regenerate at full resolution from frozen checkpoints |
| ICASSP LaTeX (4pp + 1 ref page, 2-col) | Unknown until draft read | Convert; ICASSP is much tighter than CVPR format |

## 3. Two-week execution plan

### Week 1 (Wed Aug 27 – Tue Sep 2): freeze evidence, start prose
- **D1 (Wed):** Freeze the experiment matrix (decide: keep GTSource? keep Layers2/8?).
  Launch any missing must-have runs *today* (GTSource oracle; Layers2/8 only if the story needs
  the trend line). Verify `JEPA_LQ` rerun is healthy. Every training run must end by ~Sep 2.
- **D1–2:** Build one unified evaluation script: run every frozen checkpoint over the **full**
  Deblur_MIOR validation/test split at fixed inference settings (50 steps, fixed seed), dump
  outputs to disk, compute PSNR/SSIM/LPIPS/DISTS + NIQE/MUSIQ via `calculate_metrics.py`.
  (Slurm array over checkpoints; A100 preferred, standby a30/a10 for eval-only jobs.)
- **D2–3:** Baselines: evaluate NAFNet + Restormer (train on Deblur_MIOR if no checkpoint
  exists — they're fast) and one diffusion-prior method already in-repo (OSEDiff or LucidFlux
  reproduction) with the same protocol.
- **D3–4:** Motivation analysis at dataset scale: run `analyze_jepa_svd.py` /
  `compare_jepa_features.py` over ~100–500 image pairs; produce final Fig. 1 (blur-invariance:
  cosine-sim distributions + energy-ratio per component, JEPA vs DINO vs CLIP).
- **D4–5:** Architecture figure (Stage A + Stage B, one column-width diagram).
  Write Method + the analysis section while evals run.
- **Weekend buffer:** finish stragglers; first full results table assembled.

### Week 2 (Wed Sep 3 – Tue Sep 9): write, verify, polish
- **D6–7:** Results section: main table (baselines + ours), ablation tables (encoder, layers,
  source, heads). Qualitative figure (4–5 crops × methods). Decide headline numbers.
- **D8:** Intro + related work (LucidFlux/DiffBIR line, JEPA line, restoration-with-frozen-T2I
  line) + abstract. Ensure every claim in intro maps to a table/figure.
- **D9:** Convert to ICASSP template, cut to 4 pages, references ≤1 page. Citation check.
- **D10 (Sep 8–9):** Internal review pass (simulated reviewer run + advisor pass), fix, freeze
  PDF. Keep ≥3 days of slack before the real deadline for advisor feedback / resubmission.

### Standing risks
- **Compute contention:** eval jobs for ~8 configs × full split on shared A100s — start Week 1
  Day 1, use standby queues for eval. Do not let new training displace evaluation.
- **JEPA_LQ rerun timing:** if the LQ-aligned projector is the honest headline (no GT features
  at inference), Stage-B retraining on top of it must start by ~Aug 29 to finish in time; else
  report GT-aligned projector + LQ-source ablation and be explicit about it.
- **Val-set size / variance:** past experience (Klein distill) showed tiny val splits give ±3 dB
  noise. Report on the full test split only; state N in the paper.
- **Scope control:** GoPro/standard-benchmark comparison is a *stretch goal only* — do not start
  it unless everything above is done by Sep 4.

---

## 4. Revisions after reading `papers/main.tex` (and auditing scratch)

Reading the draft changed the plan more than expected. Key findings:

### 4.1 The draft describes a different system than the one you built
`main.tex` ("JEPA-Flux") describes **I-JEPA ViT-H/14 (224px, 1280-d) + MLP projector + Flux
Kontext + RAM captioning + unsharp-mask "restore proxy" + hybrid JEPA+T5 conditioning +
full fine-tune/LoRA of the backbone**. The actual `reFlux` experiments are **V-JEPA 2.1
ViT-L/384 + DualHeadPatchCrossAttnProjector (caption-supervised alignment stage) + frozen
FLUX.2-Klein-4B + LucidFlux-style condition branch + null-prompt replacement**. The draft's own
author notes admit part of this. → **The method section needs a rewrite, not a polish.** The
restore-proxy / hybrid-text narrative belongs to an older iteration
(`models/DeblurDeform/FluxKontextJEPADeblur_model.py`); either cut it or let VL-JEPA carry the
"semantic text" axis. Retitle/reframe around V-JEPA context conditioning.

### 4.2 The draft's main tables are placeholders — treat as fabricated
Table 1 (main results), Table 2 (ablations, "200K steps"), the convergence and blur-severity
pgfplots, Table 4 (efficiency), and the appendix LoRA-rank/steps tables contain numbers no run
produced. **None of these may survive to submission.** The only real numbers in the draft are in
§4.7: the resolution-robustness probe, the two blur-invariance cosine pairs, and the recovered
ContextFluxBlurMap 5-image val score (24.44 dB). That subsection's honest-probe style is the
seed of the real analysis section.

### 4.3 Checkpoint audit (2026-08-26) — better than the draft's purge note claims
Survives on scratch with final (46/48/50K) checkpoints, ready for full-test-set eval:

| Run | Checkpoints | Usable for |
|---|---|---|
| ContextFluxBaselineJEPA (06/07) | 46–50K ✓ | main method |
| ContextFluxBaselineDINO | 46–50K ✓ | encoder ablation |
| ContextFluxBaselineNoJEPA_LoRA | 46–50K ✓ | no-context control |
| ContextFluxBaselineJEPA_Layers4 | 46–50K ✓ | depth ablation (1 vs 4) |
| ContextFluxBlurMap | 46–50K ✓ | adapter-variant comparison |
| ContextProjectorAlign_JEPA (06/19) | 16–20K ✓ | Stage-A (but see stale-path note) |
| ContextFluxBaselineCLIP | **only 30–34K** | caveat, or top-up 16K steps |
| ContextFluxBaselineVLJEPA | **only 2–6K** | needs rerun or drop |
| GTSource, Layers2, Layers8 | **never ran** | launch now or drop |
| JEPA (06/21, retrained-projector) | **purged** (logs only) | gone — rerun only if needed |

**Gotcha:** `ContextFluxBaselineJEPA.yml` points at
`ContextProjectorAlign_JEPA_2026_06_19_23_07_19/checkpoint-10000`, which no longer exists (only
16–20K survive). Fix the path before any Stage-B (re)launch.

### 4.4 Format
Draft is NeurIPS 2026 single-column style. ICASSP is its own 2-column, **4 pages + ≤1 page
references** template. Budget a real half-day for conversion + cutting; the draft is ~9 pages of
content. The TikZ overview figure also depicts the old architecture — redraw for Stage A/B.

## 5. Final merged execution plan (supersedes §3 timeline where they differ)

### Day 0–1 (Tue 8/26 – Wed 8/27) — protect and decide
1. **Back up every surviving checkpoint to depot** (`/depot/chan129/...`) — scratch already ate
   one run family; do this before anything else.
2. Fix the stale projector path in `ContextFluxBaselineJEPA.yml`.
3. Freeze the matrix. Recommended minimum for a 4-page ICASSP paper:
   encoder {JEPA, DINO, CLIP, none} + depth {1, 4} + BlurMap variant + GT-oracle.
   → Launch now: **GTSource** (high-value upper bound), **CLIP 34K→50K top-up**.
   Decide VLJEPA: rerun only if the semantic-vs-structural contrast is core; else drop.
   Drop Layers2/8 (1-vs-4 makes the point).
4. Confirm the `JEPA_LQ` ProjectorAlign rerun (started today) is healthy; if the LQ-aligned
   projector is meant to feed a Stage-B run, that Stage-B must launch by ~Fri 8/29.

### Day 2–5 (Thu 8/28 – Sun 8/31) — evidence
5. Unified eval: every surviving/finishing checkpoint over the **full** Deblur_MIOR val split,
   fixed seed + 50 steps, outputs to disk (backed up), metrics = PSNR/SSIM/LPIPS/DISTS +
   NIQE/MUSIQ via `calculate_metrics.py`. Slurm-array on standby a30/a10 so A100s keep training.
6. Baselines with identical protocol: NAFNet + Restormer (train on Deblur_MIOR — fast), and the
   in-repo diffusion baseline (OSEDiff or LucidFlux repro) if time allows.
7. Scale the motivation analysis: `compare_jepa_features.py` / `analyze_jepa_svd.py` over
   100–500 GT/blur pairs (three encoders) → Fig. 1. This also resolves the draft's excluded
   "near-zero cosine" outlier pair with real statistics.

### Day 6–9 (Mon 9/1 – Thu 9/4) — write
8. Rewrite Method around the real two-stage system; redraw the architecture figure.
9. Real main table + ablation table from step 5; qualitative figure from saved eval outputs.
10. Rewrite abstract/intro/conclusion to match actual claims (no "state-of-the-art" unless the
    table says so; the blur-invariance→conditioning-choice analysis is the defensible core).
11. Port to ICASSP template and cut to 4 pages as you go — don't convert last.

### Day 10–14 (Fri 9/5 – Tue 9/9) — verify and polish
12. Cross-check every number in the paper against the eval dumps (the placeholder-table history
    makes this audit non-optional). Citation check (draft's bib is fine but NeurIPS-styled;
    several entries need venue fixes, e.g. OSEDiff is CVPR 2024 not arXiv).
13. Internal simulated review + advisor pass; fix; freeze PDF by **Tue 9/9**, leaving buffer
    before the actual ICASSP deadline (expected ~mid-Sept — verify the exact 2027 date now).

### What to salvage from the current draft
Keep: related-work skeleton, flow-matching preliminaries (fix notation to match code), §4.7's
honest probes (expand to dataset scale), reference list (re-verify), broader-impact text.
Discard: all placeholder tables/plots, restore-proxy section, hybrid-T5 conditioning story,
old TikZ overview, LoRA appendix (contradicts `lora_finetune: false` configs).

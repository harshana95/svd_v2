# V-JEPA: which layers / components detect blurred regions (2026-10-09)

Status: DONE 2026-10-10 (GoPro, 200 held-out test crops from 11 test scenes; 400 train crops from 22 train scenes; fixed 11-frame window after a 12-frame bug was caught in review).
Code: `playground/vjepa_blur_layers/` (lib.py, extract.py, probes.py, attribute.py, followup.py, chain_k0*.sh).

## Question (user)
1. V-JEPA token difference between the *blurred-repeated* video input and the *GT video* input: which layers are affected?
2. Which part of the network is responsible for identifying the blurred area? Compare FROZEN V-JEPA vs the FINETUNED one.
3. Train a blur-region classifier on V-JEPA tokens, backprop to find activated layers (second, independent route to the same answer).

## Models
- FROZEN: `vjepa2_1_vit_large_384` (24 blocks, D=1024, patch 16, tubelet 2, 384x384 -> 24x24 grid), pretrained hub weights.
- FINETUNED: the Stage-B "vjepa_lora" predictor = the same V-JEPA with LoRA (r16, qkv/proj/fc1/fc2 in all 24 blocks) trained
  to map a blurred image repeated 10x to the V-JEPA tokens of the sharp GT video (flip-invariant loss, Adobe240 crops).
  `/scratch/gilbreth/wweligam/experiments/Motion/BlurJepaViT_stageB_vjepalora_v2_2026_09_10_14_08_22/checkpoint-50000/PeftModel_1`
  (this is the checkpoint Stage C consumes). ASSUMPTION: "finetuned final network" = this model; no V-JEPA was finetuned any other way in the repo.
  First-gen run (r8, qkv/proj only, 20k steps) also exists; used as a secondary check if time.
- RANDOM-INIT V-JEPA (same arch, no pretraining): null baseline for probes.

## Inputs (GoPro `GOPRO_Large_all`, plain mean in [0,1], window 11, vjepa window 10 -- same as Stage-B training)
Square crop (no letterbox -> no pad tokens), resized to 384. Per sample five conditions:
- `gt_vid`   : 10 real consecutive sharp frames
- `rep_sharp`: centre sharp frame x10  (static clip)
- `rep_blur` : blurred image x10      (the Stage-B / Stage-A "fake clip")
- `rep_patch`: sharp centre frame with ONE blur patch spliced in (25% area, seeded) x10 -> exact per-token blur labels
- (full-blur vs full-sharp = image-level generalisation test of the patch-trained probes)
The 2x2 split (advisor): `rep_blur` vs `rep_sharp` isolates BLUR; `rep_sharp` vs `gt_vid` isolates MOTION/TEMPORAL; `rep_blur` vs `gt_vid` is the requested
combined difference.

## Analyses
A. Per-block token difference (cosine, standardised rel-L2), frozen & finetuned, for the three pairs above; inside- vs outside-patch (rep_patch vs rep_sharp).
B. Linear probe per block (token -> blur / not blur), AUROC vs depth; scene-disjoint train/test; seam-adjacent tokens dropped; shuffled-label + random-init nulls.
C. Classifier on final tokens -> backprop: per-component (attn_l, mlp_l) attribution = sum grad . (out_blur - out_sharp) over blur tokens (graph-keeping hooks, NO detach / no_grad).
D. Causal check: activation patching (replace component output with the one from the sharp run) -> drop in classifier logit; rank-correlate with C; top-k vs random-k vs bottom-k.
E. Frozen vs finetuned: per-block movement ||ft_l - frozen_l|| on blur vs sharp input (is the change blur-specific?), LoRA delta norm per block/module.

## Known traps (from advisor / memory)
- `extract_vjepa_tokens_per_block` detaches -> useless for backprop; own hooks.
- Hard seam is a trivial edge cue -> drop tokens within 1 cell of the patch boundary; test on global blur/sharp too.
- Run on compute nodes via sbatch, never on gilbreth-fe0x. Do not co-locate on k010/k012/k006 without measuring.
- pgrep/pkill -f self-match; don't edit running .sh.

## Results (data: /scratch/gilbreth/wweligam/vjepa_blur_layers/full2; copy on depot Blur2Vid/vjepa_blur_layer_analysis_2026_10_10/)

**Standardised relative-L2 token difference per block (test, mean over tokens; lower = closer).** blocks 0 / 4 / 8 / 12 / 16 / 20 / 24(final):
- frozen blur-rep vs sharp-rep (isolates BLUR): .55 .60 .62 .60 .56 .50 .48 ; sharp-rep vs real GT video (isolates MOTION/static-clip): .62 .74 .75 .76 .77 .76 .75 ; blur-rep vs GT video (the requested diff): .60 .79 .84 .85 .83 .79 .79.
  The blur-rep-vs-GT gap is NOT mostly blur: the static-clip/motion term (sharp-rep vs GT) is as large and peaks in the same blocks 4-16. Blur-specific difference is already full-size at block 0 (the patch-embed/first block) and is spatially local: in a spliced-blur image, tokens inside the patch differ by .55-.64 across all depths, tokens outside by .01 (block 0) rising to .09-.11 by blocks 12-24 (attention leakage). Frozen difference-AUROC (inside vs outside) is 1.0 at every block, but so is a sharp-splice seam control -> difference maps show "changed", not "blurred".
- finetuned (Stage-B LoRA): rep_sharp-vs-GT gap shrinks in late blocks (.58 at b20 vs .76), and ft(blur-rep) vs the frozen GT-video tokens (its training target) is .59 at the final block vs .79 for frozen -> the LoRA learns to pull late blocks toward the GT-video tokens. Movement of ft away from frozen on a blur input is .21 (b0) -> .74 (b6-8) -> .64 (final), on a sharp input only .18 -> .48 -> .41: the change is largest at blocks 6-12 and ~1.5x larger for blurred input.
- LoRA-by-block-group (blocks 4 per group; rl2 of final tokens to frozen GT-video tokens, frozen 0.557, full LoRA 0.420): LoRA ONLY in blocks 4-7 = 0.471 (best single group), only 0-3 = 0.510, groups 8-11/12-15/16-19/20-23 alone = 0.52-0.53 (almost no help); removing LoRA from blocks 8-11 or 12-15 changes nothing (0.418-0.419), removing 0-3 or 4-7 hurts most (0.433/0.437). Per-block relative weight change is ~0.1-0.2 everywhere (largest in attn.qkv blocks 4-9), so weight-change size does not predict usefulness.

**Where blur is decodable (linear probe per block, position-matched AUROC, interior tokens >=3 cells from the seam, scene-disjoint).** blocks 0/2/4/6/8/12/16/20/24:
- frozen .79 .92 .96 .98 .99 .98 .97 .95 .93 ; finetuned .79 .92 .97 .98 .98 .99 .97 .97 .94 ; random-init V-JEPA .58-.62 flat ; shuffled-label null .47-.53.
- Caveat found in review: blur patches are centre-biased, a position-only probe scores .90 unmatched / .80 position-matched (4x4 bins). Block 0 (.79) is therefore NOT evidence of blur sensitivity; blocks >=2 clearly are. Decodability saturates by block 6-8 then DECLINES in the last third (.93-.94 at the final block), and the finetune keeps it slightly higher in late blocks (.94 vs .93; .97 vs .95 at b20). The same probes read the sharp-splice seam control as not blur (AUROC blur-vs-seam-only .91-.97) and transfer only moderately to genuinely global blur (.69-.80), weakest in blocks 8-14. Weak-blur (lowest-tercile strength) patches: .70 (b0) -> .96 (b8-14) -> .86 (final).

**Classifier + backprop attribution (head trained on final tokens, linear; linear head AUROC .98; n=80 test crops, strength>=.01).** Mean logit rise of blurred over sharp input = 11.1 (frozen) / 8.8 (finetuned).
- Backprop (grad x delta per residual writer: embed, attn_i, mlp_i) vs ground-truth necessity by activation patching: Spearman .85 (frozen) / .94 (finetuned); vs sufficiency .72 / .80; the direct-to-final-stream variant is poor (.44 / .10) -> the final-stream-only view misses most effects, use the total-effect attribution. Rank/eval halves are scene-disjoint: split-half Spearman .91/.93 (attribution), .95/.89 (patching).
- Dominant components: embed and mlp0 (patching either alone from the sharp run removes ~all of the effect: necessity ~10.8 of 11.1), trivial in the sense that blur enters the network there. First-order attribution overstates mlp0 (15.5 vs 10.8) and gives embed the wrong sign (-5.2) -> never read first-order values for the earliest components.
- Excluding block 0, top components by held-out ranking: frozen attn14, mlp2, attn22, mlp22, attn4, mlp18, attn21, attn17, mlp23; finetuned attn14, mlp2, attn9, attn10, attn4, attn20, mlp3, mlp6, attn17. attn14 and mlp2 are the top-2 in BOTH models. Patching just the top-k (ranked) from the sharp run on held-out scenes removes (frozen, of 11.6): k=2 3.3, 4 4.2, 8 5.6, 16 7.1, 32 8.8 vs random-k 0.4/0.5/1.4/3.7/6.3; finetuned (of 9.3): 2.3, 3.7, 5.0, 5.7, 6.4 vs random .1/.4/.9/3.0/5.7. Bottom-k is a worst-case, not a null (more-negative first-order attribution), ignore.
- Block-group necessity (patch four blocks' writers with the sharp values; units of logit, frozen/finetuned): blocks 0-3 11.0/8.6, 4-7 3.8/2.6, 8-11 .9/.5, 12-15 1.1/1.0, 16-19 ~0/~0, 20-23 3.0/1.0. Suffix patching (all writers in blocks >=b from sharp; of 11.6/9.3): b=1 8.8/7.9, 4 8.0/7.5, 8 5.8/2.6, 12 3.7/2.7, 16 2.9/1.7, 20 2.9/1.2: roughly 75% of the effect needs blocks >=1, half needs blocks >=8, ~25% needs blocks >=16 (frozen) and ~12% (finetuned).
- Reading: the blur signal is injected in block 0 (+ mlp2/3), carried by the residual stream, and then (a) amplified by mid-block attention (attn4, attn9/10, attn14) and (b) frozen re-uses it again in the last 4 blocks (attn21-22, mlp22-23), which the finetune largely drops (necessity 3.0 -> 1.0). Individually, single components after block 3 matter little (redundancy via the residual stream); the layer story is only visible at group level.

**Answers to the three questions.** (1) Layers affected: the blur-vs-sharp token difference is full-size from block 0 on and stays local to the blurred region (leakage outside grows with depth); the blur-rep vs GT-video difference peaks at blocks 6-16 but is dominated by the static-clip (motion) term. (2) Responsible part: patch-embed/block-0 MLP inject it, blocks 4-7 and 14 (attention) carry/amplify it, and decodability saturates at blocks 6-8; finetuning concentrates its useful change in blocks 0-7 and reduces late-block dependence. (3) The classifier-backprop route (Spearman .85-.94 with causal patching, top-k patching 3-9x random-k) reproduces the same early/mid-block localisation as the probes, so the two independent routes agree.

## Caveats
- Synthetic spliced blur on GoPro with plain frame-mean blur; global-blur transfer of the probes is only .69-.80 AUROC, so conclusions about natural spatially-varying blur are weaker than the spliced-patch numbers suggest. Position confound only partially removed (.80 position-only matched AUROC); block-0 AUROC is not evidence.
- Linear heads/probes only. Attribution/patching target is the classifier logit, not Wan's actual use of the tokens (Wan reads V-JEPA tokens through cross-attention; the layers that matter for decodability need not be the layers that matter for Wan).
- The "finetuned" model is the Stage-B LoRA (maps blur-rep to GT-video tokens, trained on Adobe240), not a model finetuned for blur localisation.
- 80 attribution crops from 11 test scenes; the prefix-patching result in followup json is uninformative (prefix included embed) and was not used.
- A 12-frame window bug (found by a code-review agent) affected the first pass; all numbers above are from the fixed rerun. First-pass conclusions were unchanged.

## Figures (`figs/` next to this doc on depot; code `figs_agg.py`, `figs_qual.py`)
- fig1 token difference per block (blur-only / static-clip / requested), fig2 inside vs outside the patch, fig3 probes (interior, seam control, whole-frame transfer; random-init and position-only references), fig4 per-component attribution and patching bars, fig5 attribution-vs-patching scatter + top-k vs random-k + suffix + group patching, fig6 LoRA weight change / group ablation / movement.
- fig7 (4 test crops, spliced patch): token-difference maps (frozen, blocks 0/8/16/final) stay inside the patch with a faint halo growing with depth; probe maps are noisy at block 2, clean rectangles at block 8, slightly noisier at the final block; the finetuned final-block map is similar to frozen.
- fig8 (same crops, whole-frame blur): blur-rep vs GT-video difference maps are spatially diffuse and nearly the same as the sharp-rep vs GT-video (static-clip) maps, so no blur-specific localisation is visible there; the block-8 probe fires on broad regions that only loosely track the pixel |blur-sharp| map (best on the strongly blurred crop, poor on the weakly blurred ones), matching the weak whole-frame transfer (AUROC .70-.75 at block 8).
- Component bars also show negative writers (attn11 in both models, mlp16 in frozen): patching them to sharp RAISES the blur logit, i.e. they suppress the blur signal.
- Palette: Okabe-Ito blue/vermillion/grey; the dataviz palette validator could not be run (no node on the cluster).

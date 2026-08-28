"""Compare PSNR/MSE across this session's Flux2KleinDistill(VJEPA) checkpoints
on the shared Deblur_MIOR validation split, bypassing the trackers (comet/tb)
so the numbers print directly instead of only landing in a dashboard.
"""
import gc
import logging
import os

import numpy as np
import torch
import yaml
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from models import create_model
from utils.misc import DictAsMember, calculate_mse, calculate_psnr

logging.basicConfig(level=logging.INFO)
logger = get_logger("eval")

CHECKPOINTS = [
    dict(
        # K=1 warmup checkpoint that DINO_adversarial_K4 (below) resumes
        # from — confirmed never itself resumed (fresh LoRA, continuous
        # run), so unaffected by the PEFT-resume bug regardless of fix date.
        # Added 2026-08-13 as a baseline for judging the postfix K4 re-run,
        # mirroring Hybrid_ReflowLoss_DINO_K1's role for the Hybrid line.
        label="DINO_K1",
        config="checkpoints/reFlux/Flux2KleinDistill.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinDistill_2026_07_24_01_49_39",
        checkpoint="checkpoint-50000",
        num_inference_steps=1,
        full_res_supported=True,
    ),
    dict(
        # PRE-FIX, SUSPECT (as of 2026-08-13): resumed from DINO_K1 above on
        # 2026-07-25, before the base_model.py PeftModel-resume fix
        # (2026-08-01) — per peft_resume_optimizer_disconnect memory note,
        # this class of resume silently never trained past its source
        # checkpoint. Kept here for reference; see DINO_adversarial_K4_postfix
        # below for the re-verified version.
        label="DINO_adversarial_K4",
        config="checkpoints/reFlux/Flux2KleinDistill.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinDistill_2026_07_25_16_50_15",
        checkpoint="checkpoint-70000",
        num_inference_steps=4,
        full_res_supported=True,
    ),
    dict(
        # PRE-FIX, SUSPECT — resumed from DINO_adversarial_K4 above (itself
        # suspect), so this is doubly so; confirmed byte-identical to it in
        # the original session (see peft_resume_optimizer_disconnect memory
        # note). No postfix re-run planned yet — lower priority than
        # verifying the primary K4 checkpoints first.
        label="DINO_fidelity_only_K4",
        config="checkpoints/reFlux/Flux2KleinDistill.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinDistill_2026_07_27_23_10_06",
        checkpoint="checkpoint-90000",
        num_inference_steps=4,
        full_res_supported=True,
    ),
    dict(
        # PRE-FIX, SUSPECT — resumed from Flux2KleinDistillVJEPA_2026_07_26_
        # 17_45_23/checkpoint-20000 (the VJEPA K=1 warmup, itself a fresh
        # continuous run so unaffected) on 2026-07-27, before the fix.
        # Confirmed byte-identical to its K=1 source in the original session.
        # See VJEPA_condition_K4_postfix below for the re-verified version.
        label="VJEPA_condition_K4",
        config="checkpoints/reFlux/Flux2KleinDistillVJEPA.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinDistillVJEPA_2026_07_27_02_41_27",
        checkpoint="checkpoint-40000",
        num_inference_steps=4,
        full_res_supported=True,
    ),
    dict(
        label="Reflow_GAN_K2",
        config="checkpoints/reFlux/Flux2KleinReflow.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinReflow_2026_07_20_01_11_33",
        checkpoint="checkpoint-50000",
        num_inference_steps=2,
        # Flux2KleinReflow_model's validate_step wasn't given patchwise
        # support (only Flux2KleinDistill_model was) -- val.patched: true
        # would crash it the same way Flux2KleinDistill_model used to.
        full_res_supported=False,
    ),
    dict(
        label="Hybrid_ReflowLoss_DINO_K1",
        config="checkpoints/reFlux/Flux2KleinHybrid.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinHybrid_2026_07_29_02_30_28",
        checkpoint="checkpoint-20000",
        num_inference_steps=1,
        # Flux2KleinHybrid_model.validate_step was ported from
        # Flux2KleinDistill_model.py and keeps its patched branch.
        full_res_supported=True,
    ),
    dict(
        label="Hybrid_ReflowLoss_DINO_K4_lr2e4",
        config="checkpoints/reFlux/Flux2KleinHybrid.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinHybrid_2026_07_30_02_35_12",
        checkpoint="checkpoint-40000",
        num_inference_steps=4,
        full_res_supported=True,
    ),
    # Hybrid_ReflowLoss_DINO_K4_lr2e4_replicate2 dropped: turned out to be a
    # bit-identical deterministic re-execution (fixed seed + restored RNG
    # state from the resumed checkpoint), not an independent statistical
    # trial -- see klein_reflow_baseline_and_hybrid memory note. No longer
    # useful to keep re-evaluating.
    dict(
        # Same resume checkpoint/K=4 as Hybrid_ReflowLoss_DINO_K4_lr2e4, but
        # with the generator LR reverted to the original 5e-5 (that run's 4x
        # bump to 2e-4 caused a reproducible 2.4-3.3dB PSNR regression despite
        # healthy-looking training losses -- see memory note). Tests whether
        # K=4 itself hurts this recipe, or whether it was specifically the LR.
        label="Hybrid_ReflowLoss_DINO_K4_lr5e5",
        config="checkpoints/reFlux/Flux2KleinHybrid.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinHybrid_2026_07_31_05_25_31",
        checkpoint="checkpoint-40000",
        num_inference_steps=4,
        full_res_supported=True,
    ),
    dict(
        # Same resume lineage/K=4/lr=5e-5 as Hybrid_ReflowLoss_DINO_K4_lr5e5
        # above, but trained after the base_model.py PeftModel-resume fix
        # (see peft_resume_optimizer_disconnect memory note) -- every prior
        # "K=4" Hybrid checkpoint (lr2e4 and lr5e5 above) turned out to be
        # byte-identical to the K=1 checkpoint it resumed from, so those
        # numbers never actually reflected K=4 training. This is the first
        # Hybrid K=4 checkpoint where the optimizer was verifiably connected
        # across the resume. Resumed from
        # Flux2KleinHybrid_2026_08_01_06_42_11/checkpoint-26000 (itself a
        # continuation of this same postfix lineage, interrupted by a 7-day
        # walltime timeout, not a training failure) and completed cleanly to
        # checkpoint-40000 on 2026-08-06 (17h33m for the last 14k steps,
        # loss_all 1.4->0.91, no NaN/collapse).
        label="Hybrid_ReflowLoss_DINO_K4_lr5e5_postfix",
        config="checkpoints/reFlux/Flux2KleinHybrid.yml",
        resume_from_path="/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinHybrid_2026_08_05_12_06_53",
        checkpoint="checkpoint-40000",
        num_inference_steps=4,
        full_res_supported=True,
    ),
]


def load_opt(config_path, resume_from_path, checkpoint, full_res=False):
    with open(config_path) as f:
        data = OmegaConf.load(f)
        data = OmegaConf.to_yaml(data, resolve=True)
        data = yaml.safe_load(data)
    opt = DictAsMember(**data)
    opt.is_train = False
    opt.name = "eval"
    opt.tracker_project_name = "eval"
    opt.report_to = "tensorboard"  # avoid spawning a new comet_ml experiment for a quick local eval
    opt.comment = ""
    opt.test = False
    opt.opt_path = config_path
    opt.path.resume_from_path = resume_from_path
    opt.path.resume_from_checkpoint = checkpoint
    if full_res:
        # Native-resolution (1920x1080) images instead of the 512x512 crop
        # datasets.val normally applies, so grids()/setup_patches() in
        # validate_step actually has something larger than one patch to
        # tile. patch_size_h/w/overlap/max_minibatch are left at whatever
        # the config already has (val.patched is the only flip needed).
        opt.datasets.val.transforms.pop("crop", None)
        opt.val.patched = True
        # datasets.val doesn't set num_proc, so hf_dataset.py's .map() call
        # defaults to 4 -- forking worker processes *after* the model this
        # script already loaded is on the GPU, which deadlocks on an
        # inherited lock from a background CUDA/thread-pool thread (see
        # hf_dataset.py's own "if stuck use accelerate launch instead of
        # python" comment, and Flux2KleinReflowT2I.yml's datasets.train,
        # which sidesteps the same hazard with num_proc: 1). The 512-crop
        # pass never hits this because Flux2KleinDistill_model.py's own
        # setup_dataloaders builds the val dataloader with no `transforms:`
        # blowup risk at 512x512 -- forcing single-process here only for
        # the larger full-res .map() call.
        opt.datasets.val.num_proc = 1
    return opt


def resume_weights(model, resume_from_path, checkpoint):
    load_path = os.path.join(resume_from_path, checkpoint)
    model.accelerator.print(f"Resuming from checkpoint {load_path}")
    model.accelerator.load_state(load_path)
    model._sanitize_optimizer_state_shapes()
    model.load_other_parameters(load_path)
    model.after_resume_checkpoint(load_path)


@torch.no_grad()
def evaluate(model, num_inference_steps):
    from utils.dataset_utils import merge_patches
    import einops

    gt_key = model.test_dataloader.dataset.gt_key
    lq_key = model.test_dataloader.dataset.lq_key
    patched = bool(model.opt.val.patched)
    for m in model.models:
        m.eval()

    psnrs, mses = [], []
    for batch in model.test_dataloader:
        model.feed_data(batch, is_train=False)

        if patched:
            # Mirrors Flux2KleinDistill_model.validate_step's patched branch:
            # feed_data's grids() call already patched lq/gt and stashed the
            # pre-patch shape in model.original_size.
            b, c, h, w = model.original_size[lq_key]
            pred = []
            for _ in model.setup_patches():
                patch = model.sample[lq_key]
                if patch.shape[1] == 1:
                    patch = patch.repeat(1, 3, 1, 1)
                pred.append(model.forwardpass(patch, num_steps=num_inference_steps))
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, "(b n) c h w -> b n c h w", b=b)
            out = torch.stack([
                merge_patches(pred[i], model.sample[lq_key + "_patched_pos"])[..., :h, :w]
                for i in range(len(pred))
            ])
            gt = model.sample[gt_key + "_original"]
        else:
            lq = model.sample[lq_key]
            gt = model.sample[gt_key]
            if lq.shape[1] == 1:
                lq = lq.repeat(1, 3, 1, 1)
            if gt.shape[1] == 1:
                gt = gt.repeat(1, 3, 1, 1)
            out = model.forwardpass(lq, num_steps=num_inference_steps)

        gt_np = (gt.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        for i in range(gt_np.shape[0]):
            gt_hwc = np.transpose(gt_np[i], (1, 2, 0))
            out_hwc = np.transpose(out_np[i], (1, 2, 0))
            psnrs.append(float(calculate_psnr(out_hwc, gt_hwc, crop=30)))
            mses.append(float(calculate_mse(out_hwc, gt_hwc, crop=30)))

    return dict(
        psnr_mean=float(np.mean(psnrs)),
        psnr_std=float(np.std(psnrs)),
        mse_mean=float(np.mean(mses)),
        n=len(psnrs),
        psnrs=psnrs,
    )


def run_pass(pass_label, full_res):
    results = {}
    for spec in CHECKPOINTS:
        if full_res and not spec.get("full_res_supported", False):
            continue
        print(f"\n{'='*20} [{pass_label}] {spec['label']} {'='*20}", flush=True)
        try:
            opt = load_opt(spec["config"], spec["resume_from_path"], spec["checkpoint"], full_res=full_res)
            model = create_model(opt, logger)
            model.setup_dataloaders()
            model.prepare()
            resume_weights(model, spec["resume_from_path"], spec["checkpoint"])
            metrics = evaluate(model, spec["num_inference_steps"])
            print(f"RESULT [{pass_label}] {spec['label']}: {metrics}", flush=True)
            results[spec["label"]] = metrics
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            import traceback
            print(f"FAILED [{pass_label}] {spec['label']}:", flush=True)
            traceback.print_exc()

    print(f"\n{'='*20} SUMMARY [{pass_label}] {'='*20}", flush=True)
    for label, metrics in results.items():
        print(
            f"{label:30s}  PSNR={metrics['psnr_mean']:.2f}+-{metrics['psnr_std']:.2f} dB  "
            f"MSE={metrics['mse_mean']:.5f}  n={metrics['n']}",
            flush=True,
        )
    return results


def main():
    run_pass("512-crop", full_res=False)
    run_pass("full-res-patched", full_res=True)


if __name__ == "__main__":
    main()

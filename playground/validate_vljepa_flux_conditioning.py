#!/usr/bin/env python3
"""Quick GPU validation for ContextFluxBaseline with VL-JEPA semantic conditioning."""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from omegaconf import OmegaConf

from utils.misc import DictAsMember


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-opt",
        default="checkpoints/reFlux/ContextFluxBaseline.yml",
        help="Training config YAML",
    )
    p.add_argument("--max-train-steps", type=int, default=2)
    p.add_argument("--max-val-steps", type=int, default=1)
    p.add_argument("--num-inference-steps", type=int, default=5)
    p.add_argument(
        "--projector-only",
        action="store_true",
        help="Run shape checks only (no gated HF model downloads).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from accelerate.logging import get_logger
    from models import create_model

    logger = get_logger(__name__)

    with open(args.opt) as f:
        data = yaml.safe_load(OmegaConf.to_yaml(OmegaConf.load(f), resolve=True))
    opt = DictAsMember(**data)
    opt.test = True
    opt["name"] = "validate_vljepa_flux_conditioning"
    opt["tracker_project_name"] = "validate"
    opt.train.max_train_steps = args.max_train_steps
    opt.val.max_val_steps = args.max_val_steps
    opt.val.num_inference_steps = args.num_inference_steps
    opt.opt_path = args.opt

    print("=" * 60)
    print("ContextFluxBaseline VL-JEPA validation")
    print(f"  config: {args.opt}")
    print(f"  use_vljepa_condition: {getattr(opt, 'use_vljepa_condition', False)}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    checks = []

    def ok(name):
        checks.append((name, True))
        print(f"[PASS] {name}")

    def fail(name, err):
        checks.append((name, False))
        print(f"[FAIL] {name}: {err}")

    if args.projector_only:
        from models.reFlux.dino_helpers import get_dino_num_patches
        from models.LucidFlux.jepa_helpers import get_vjepa_num_patches
        from models.reFlux.vljepa_helpers import VLJEPA_EMBED_DIM, preprocess_vljepa_image
        from models.reFlux.cross_attn_projector import PatchCrossAttnProjector, VLJEPAProjector

        seq_dim = 15360
        vljepa_proj = VLJEPAProjector(vljepa_dim=VLJEPA_EMBED_DIM, seq_dim=seq_dim, num_slots=512)
        s_y = torch.randn(2, VLJEPA_EMBED_DIM)
        out = vljepa_proj(s_y)
        assert out.shape == (2, 512, seq_dim), f"VL-JEPA shape mismatch: {tuple(out.shape)}"

        jepa_num_slots = get_vjepa_num_patches("vjepa2_1_vit_large_384", 384)
        jepa_proj = PatchCrossAttnProjector(
            embed_dim=1024, seq_dim=seq_dim, num_slots=jepa_num_slots
        )
        jepa_tokens = torch.randn(2, jepa_num_slots, 1024)
        jepa_out = jepa_proj(jepa_tokens)
        assert jepa_out.shape == (2, jepa_num_slots, seq_dim), (
            f"V-JEPA shape mismatch: {tuple(jepa_out.shape)}"
        )

        dino_num_slots = get_dino_num_patches("facebook/dinov2-large", 518)
        dino_proj = PatchCrossAttnProjector(
            embed_dim=1024, seq_dim=seq_dim, num_slots=dino_num_slots
        )
        dino_tokens = torch.randn(2, dino_num_slots, 1024)
        dino_out = dino_proj(dino_tokens)
        assert dino_out.shape == (2, dino_num_slots, seq_dim), (
            f"DINO shape mismatch: {tuple(dino_out.shape)}"
        )

        x = torch.rand(2, 3, 512, 512)
        pv = preprocess_vljepa_image(x, num_frames=1, image_size=256)
        assert pv.shape == (2, 1, 3, 256, 256)
        print("[PASS] projector-only shape checks (VL-JEPA, V-JEPA, DINO)")
        return 0

    try:
        model = create_model(opt, logger)
        ok("create_model")
    except Exception as e:
        fail("create_model", e)
        traceback.print_exc()
        return 1

    if getattr(model, "vljepa_model", None) is not None:
        ok("vljepa_model loaded")
    else:
        fail("vljepa_model loaded", "vljepa_model is None")
        return 1

    try:
        model.setup_dataloaders()
        ok("setup_dataloaders")
    except Exception as e:
        fail("setup_dataloaders", e)
        traceback.print_exc()
        return 1

    try:
        model.prepare()
        ok("prepare")
    except Exception as e:
        fail("prepare", e)
        traceback.print_exc()
        return 1

    try:
        batch = next(iter(model.dataloader))
        with model.accelerator.accumulate(*model.models):
            model.feed_data(batch, is_train=True)
            losses = model.optimize_parameters()
        ok(f"optimize_parameters (loss={float(losses['all']):.6f})")
    except Exception as e:
        fail("optimize_parameters", e)
        traceback.print_exc()
        return 1

    try:
        val_iter = iter(model.test_dataloader)
        vbatch = next(val_iter)
        gt_key = model.test_dataloader.dataset.gt_key
        lq_key = model.test_dataloader.dataset.lq_key
        model.validate_step(vbatch, 0, lq_key, gt_key)
        ok("validate_step (1 step)")
    except Exception as e:
        fail("validate_step", e)
        traceback.print_exc()
        return 1

    print("=" * 60)
    passed = sum(1 for _, passed_flag in checks if passed_flag)
    print(f"All checks passed ({passed}/{len(checks)})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

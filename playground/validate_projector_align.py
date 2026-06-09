#!/usr/bin/env python3
"""Smoke test for ContextProjectorAlign (projector -> Flux VAE token alignment)."""
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
        default="checkpoints/Flux/ContextProjectorAlign_JEPA.yml",
        help="Training config YAML",
    )
    p.add_argument("--max-train-steps", type=int, default=2)
    p.add_argument("--max-val-steps", type=int, default=1)
    p.add_argument(
        "--projector-only",
        action="store_true",
        help="Run shape checks only (no HF model downloads).",
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
    opt["name"] = "validate_projector_align"
    opt["tracker_project_name"] = "validate"
    opt.train.max_train_steps = args.max_train_steps
    opt.val.max_val_steps = args.max_val_steps
    opt.opt_path = args.opt

    print("=" * 60)
    print("ContextProjectorAlign validation")
    print(f"  config: {args.opt}")
    print(f"  source_image: {getattr(opt, 'source_image', 'gt')}")
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
        from models.LucidFlux.cross_attn_projector import PatchCrossAttnProjector
        from models.LucidFlux.dino_helpers import get_dino_num_patches
        from models.LucidFlux.jepa_helpers import get_vjepa_num_patches
        from models.LucidFlux.vljepa_helpers import VLJEPA_EMBED_DIM

        crop_size = int(getattr(opt, "train_crop_size", 512))
        latent_spatial_divisor = 16
        num_slots = (crop_size // latent_spatial_divisor) ** 2
        token_dim = 128

        jepa_num_patches = get_vjepa_num_patches("vjepa2_1_vit_large_384", 384)
        jepa_proj = PatchCrossAttnProjector(
            embed_dim=1024, seq_dim=token_dim, num_slots=num_slots
        )
        jepa_tokens = torch.randn(2, jepa_num_patches, 1024)
        jepa_out = jepa_proj(jepa_tokens)
        assert jepa_out.shape == (2, num_slots, token_dim), (
            f"V-JEPA shape mismatch: {tuple(jepa_out.shape)}"
        )

        dino_num_patches = get_dino_num_patches("facebook/dinov2-large", 518)
        dino_proj = PatchCrossAttnProjector(
            embed_dim=1024, seq_dim=token_dim, num_slots=num_slots
        )
        dino_tokens = torch.randn(2, dino_num_patches, 1024)
        dino_out = dino_proj(dino_tokens)
        assert dino_out.shape == (2, num_slots, token_dim), (
            f"DINO shape mismatch: {tuple(dino_out.shape)}"
        )

        vljepa_proj = PatchCrossAttnProjector(
            embed_dim=VLJEPA_EMBED_DIM, seq_dim=token_dim, num_slots=num_slots
        )
        s_y = torch.randn(2, VLJEPA_EMBED_DIM)
        vljepa_out = vljepa_proj(s_y)
        assert vljepa_out.shape == (2, num_slots, token_dim), (
            f"VL-JEPA shape mismatch: {tuple(vljepa_out.shape)}"
        )

        target = torch.randn(2, num_slots, token_dim)
        pred_before = vljepa_proj(s_y).detach()
        mse_before = torch.nn.functional.mse_loss(pred_before, target).item()
        opt_fn = torch.optim.Adam(vljepa_proj.parameters(), lr=1e-3)
        for _ in range(5):
            opt_fn.zero_grad()
            pred = vljepa_proj(s_y)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            opt_fn.step()
        mse_after = torch.nn.functional.mse_loss(vljepa_proj(s_y).detach(), target).item()
        assert mse_after < mse_before, (
            f"MSE did not decrease: before={mse_before:.4f} after={mse_after:.4f}"
        )

        print(f"[PASS] projector-only shape checks (num_slots={num_slots}, token_dim={token_dim})")
        print(f"[PASS] MSE decreased over 5 steps: {mse_before:.4f} -> {mse_after:.4f}")
        return

    try:
        model = create_model(opt, logger)
        ok("model created")
    except Exception as exc:
        fail("model created", exc)
        traceback.print_exc()
        sys.exit(1)

    try:
        model.setup_dataloaders()
        model.prepare()
        ok("dataloaders and prepare")
    except Exception as exc:
        fail("dataloaders and prepare", exc)
        traceback.print_exc()
        sys.exit(1)

    try:
        model.train()
        ok("training loop completed")
    except Exception as exc:
        fail("training loop completed", exc)
        traceback.print_exc()
        sys.exit(1)

    failed = [name for name, passed in checks if not passed]
    if failed:
        print(f"\nFailed checks: {failed}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

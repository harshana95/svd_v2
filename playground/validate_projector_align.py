#!/usr/bin/env python3
"""Smoke test for ContextProjectorAlign (dual-head projector alignment)."""
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
        default="checkpoints/reFlux/ContextProjectorAlign_JEPA.yml",
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
        from models.reFlux.cross_attn_projector import (
            DualHeadPatchCrossAttnProjector,
            PatchCrossAttnProjector,
            VaeToTextTargetBridge,
            build_dual_head_projector,
            load_jepa_projector_weights,
            resample_text_tokens,
        )
        from models.reFlux.dino_helpers import get_dino_num_patches
        from models.LucidFlux.helpers import pack_projector_vae_tokens
        from models.LucidFlux.jepa_helpers import get_vjepa_num_patches
        from models.reFlux.vljepa_helpers import VLJEPA_EMBED_DIM

        text_dim = 7680
        lat_h = lat_w = 32
        vae_slots = lat_h * lat_w
        vjepa_num_slots = get_vjepa_num_patches("vjepa2_1_vit_large_384", 384)
        vjepa_grid = int(vjepa_num_slots**0.5)

        dual_proj = build_dual_head_projector(
            embed_dim=1024,
            text_seq_dim=text_dim,
            text_num_slots=vjepa_num_slots,
            vae_seq_dim=128,
            vae_num_slots=vae_slots,
        )
        jepa_tokens = torch.randn(2, vjepa_num_slots, 1024)
        dual_out = dual_proj(jepa_tokens, heads="both")
        assert dual_out["text"].shape == (2, vjepa_num_slots, text_dim), (
            f"Text head shape mismatch: {tuple(dual_out['text'].shape)}"
        )
        assert dual_out["vae"].shape == (2, vae_slots, 128), (
            f"VAE head shape mismatch: {tuple(dual_out['vae'].shape)}"
        )
        text_only = dual_proj(jepa_tokens, heads="text")
        assert text_only.shape == (2, vjepa_num_slots, text_dim)
        vae_only = dual_proj(jepa_tokens, heads="vae")
        assert vae_only.shape == (2, vae_slots, 128)

        flux_text = torch.randn(2, 512, text_dim)
        resampled = resample_text_tokens(flux_text, vjepa_num_slots)
        assert resampled.shape == (2, vjepa_num_slots, text_dim), (
            f"Resample shape mismatch: {tuple(resampled.shape)}"
        )

        class _MockPipe:
            @staticmethod
            def _prepare_image_ids(latents):
                return torch.zeros(1, latents[0].shape[2] * latents[0].shape[3], 3)

        vae_latents, vae_ids = pack_projector_vae_tokens(
            dual_out["vae"],
            _MockPipe,
            lat_h,
            lat_w,
            128,
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert vae_latents.shape == (2, vae_slots, 128)
        assert vae_ids.shape[0] == 2

        bridge = VaeToTextTargetBridge(
            latent_token_dim=128,
            text_dim=text_dim,
            lat_h=lat_h,
            lat_w=lat_w,
            target_h=vjepa_grid,
            target_w=vjepa_grid,
        )
        vae_tokens = torch.randn(2, vae_slots, 128)
        bridge_out = bridge(vae_tokens)
        assert bridge_out.shape == (2, vjepa_num_slots, text_dim), (
            f"Bridge shape mismatch: {tuple(bridge_out.shape)}"
        )

        dino_num_patches = get_dino_num_patches("facebook/dinov2-large", 518)
        dino_dual = build_dual_head_projector(
            embed_dim=1024,
            text_seq_dim=text_dim,
            text_num_slots=dino_num_patches,
            vae_seq_dim=128,
            vae_num_slots=vae_slots,
        )
        dino_tokens = torch.randn(2, dino_num_patches, 1024)
        dino_out = dino_dual(dino_tokens, heads="both")
        assert dino_out["text"].shape == (2, dino_num_patches, text_dim)

        vljepa_num_slots = 512
        vljepa_proj = PatchCrossAttnProjector(
            embed_dim=VLJEPA_EMBED_DIM,
            seq_dim=text_dim,
            num_slots=vljepa_num_slots,
        )
        s_y = torch.randn(2, VLJEPA_EMBED_DIM)
        vljepa_out = vljepa_proj(s_y)
        assert vljepa_out.shape == (2, vljepa_num_slots, text_dim), (
            f"VL-JEPA shape mismatch: {tuple(vljepa_out.shape)}"
        )

        target_text = torch.randn(2, vjepa_num_slots, text_dim)
        target_vae = torch.randn(2, vae_slots, 128)
        pred_before = dual_proj(jepa_tokens, heads="both")
        mse_before = (
            torch.nn.functional.mse_loss(pred_before["text"], target_text)
            + torch.nn.functional.mse_loss(pred_before["vae"], target_vae)
        ).item()
        opt_fn = torch.optim.Adam(dual_proj.parameters(), lr=1e-3)
        for _ in range(5):
            opt_fn.zero_grad()
            pred = dual_proj(jepa_tokens, heads="both")
            loss = torch.nn.functional.mse_loss(pred["text"], target_text) + torch.nn.functional.mse_loss(
                pred["vae"], target_vae
            )
            loss.backward()
            opt_fn.step()
        pred_after = dual_proj(jepa_tokens, heads="both")
        mse_after = (
            torch.nn.functional.mse_loss(pred_after["text"], target_text)
            + torch.nn.functional.mse_loss(pred_after["vae"], target_vae)
        ).item()
        assert mse_after < mse_before, (
            f"MSE did not decrease: before={mse_before:.4f} after={mse_after:.4f}"
        )

        ckpt_path = "/tmp/validate_dual_projector.bin"
        torch.save(dual_proj.state_dict(), ckpt_path)
        reload_proj = build_dual_head_projector(
            embed_dim=1024,
            text_seq_dim=text_dim,
            text_num_slots=vjepa_num_slots,
            vae_seq_dim=128,
            vae_num_slots=vae_slots,
        )
        load_jepa_projector_weights(reload_proj, ckpt_path)
        baseline_reload = build_dual_head_projector(
            embed_dim=1024,
            text_seq_dim=text_dim,
            text_num_slots=vjepa_num_slots,
            vae_seq_dim=128,
            vae_num_slots=vae_slots,
        )
        load_jepa_projector_weights(baseline_reload, ckpt_path)
        os.remove(ckpt_path)

        print(
            f"[PASS] projector-only dual-head shape checks "
            f"(text_slots={vjepa_num_slots}, vae_slots={vae_slots}, text_dim={text_dim})"
        )
        print(f"[PASS] MSE decreased over 5 steps: {mse_before:.4f} -> {mse_after:.4f}")
        print("[PASS] strict hot-plug reload into align + baseline dual projector")
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

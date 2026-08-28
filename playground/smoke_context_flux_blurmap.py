#!/usr/bin/env python3
"""A100 smoke test for ContextFluxBlurMap adapters + one FM train step.

Usage on an A100 node / via sbatch:
  python playground/smoke_context_flux_blurmap.py
  python playground/smoke_context_flux_blurmap.py --full
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-opt", default="checkpoints/reFlux/ContextFluxBlurMap.yml")
    p.add_argument(
        "--full",
        action="store_true",
        help="Load Flux2 + V-JEPA and run one train + val step",
    )
    return p.parse_args()


def test_adapters():
    import torch
    from models.reFlux.blur_adapters import (
        BlurMapExtractor,
        BlurResidualCoupler,
        JEPAPerceptualEncoder,
        PCMModule,
        PGAModule,
        TimestepLayerModulation,
        map_to_tokens,
        tokens_to_grid,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[adapters] device={device}")
    if device == "cuda":
        print(f"[adapters] gpu={torch.cuda.get_device_name(0)}")

    ext = BlurMapExtractor(blur_dim=16, global_dim=16, num_bands=8).to(device)
    x = torch.rand(2, 3, 64, 64, device=device)
    feats = ext(x)
    assert feats.blur_map.shape == (2, 16, 64, 64), feats.blur_map.shape
    tok = map_to_tokens(feats.blur_map, 8, 8)
    assert tok.shape == (2, 64, 16), tok.shape

    conn = JEPAPerceptualEncoder(1024, 128).to(device)
    jepa = torch.randn(2, 256, 1024, device=device)
    perc = tokens_to_grid(conn(jepa), 8, 8)
    assert perc.shape == (2, 64, 128), perc.shape

    mod = TimestepLayerModulation(4).to(device)
    m = mod(torch.rand(2, device=device), 0)
    coupler = BlurResidualCoupler(128, 16, 32).to(device)
    pcm = PCMModule(128, 32).to(device)
    h = torch.randn(2, 64, 128, device=device)
    h2 = pcm(h, perc, m)
    h3 = coupler(h2, tok, perc, m)
    assert h3.shape == h.shape

    pga = PGAModule(
        128,
        heads=4,
        head_dim=32,
        blur_dim=16,
        global_dim=16,
        num_bands=8,
        rank=4,
    ).to(device)
    dv = pga.value_residual(
        h, m, feats.blur_map.mean([2, 3]), feats.g, feats.r
    )
    assert dv.shape == h.shape
    print("[PASS] adapter shape checks")


def test_full(opt_path: str):
    import torch
    import yaml
    from omegaconf import OmegaConf
    from accelerate.logging import get_logger
    from utils.misc import DictAsMember
    from models import create_model

    assert torch.cuda.is_available(), "CUDA required for --full"
    print(f"[full] gpu={torch.cuda.get_device_name(0)}")

    with open(opt_path) as f:
        data = yaml.safe_load(OmegaConf.to_yaml(OmegaConf.load(f), resolve=True))
    opt = DictAsMember(**data)
    opt.test = True
    opt["name"] = "smoke_context_flux_blurmap"
    opt["tracker_project_name"] = "smoke"
    opt.report_to = "tensorboard"
    opt.num_gpu = 1
    opt.train.max_train_steps = 2
    opt.train.batch_size = 1
    opt.val.max_val_steps = 1
    opt.val.num_inference_steps = 4
    opt.opt_path = opt_path

    logger = get_logger(__name__)
    model = create_model(opt, logger)
    print("[PASS] create_model")

    model.setup_dataloaders()
    print("[PASS] setup_dataloaders")
    model.prepare()
    print("[PASS] prepare")

    batch = next(iter(model.dataloader))
    with model.accelerator.accumulate(*model.models):
        model.feed_data(batch, is_train=True)
        losses = model.optimize_parameters()
    print(f"[PASS] optimize_parameters (loss={float(losses['all'].detach()):.6f})")

    vbatch = next(iter(model.test_dataloader))
    model.validate_step(
        vbatch,
        0,
        model.test_dataloader.dataset.lq_key,
        model.test_dataloader.dataset.gt_key,
    )
    print("[PASS] validate_step")


def main():
    args = parse_args()
    print("=" * 60)
    print("ContextFluxBlurMap smoke test")
    print("=" * 60)
    try:
        test_adapters()
        if args.full:
            test_full(args.opt)
        print("ALL PASS")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

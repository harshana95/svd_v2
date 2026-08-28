#!/usr/bin/env python
"""Offline smoke test -- validates LumaFlux modules + the full FLUX wiring on a
*tiny* randomly-initialised FLUX (no 12B download, runs on CPU).

Run:  python smoke_test.py
"""

from __future__ import annotations

import torch

from lumaflux import color
from lumaflux.rqs import monotone_rqs, RQSToneDecoder
from lumaflux.model import LumaFlux, LumaFluxConfig
from lumaflux.losses import LumaFluxLoss
from lumaflux.dataset import SyntheticPairDataset


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_color():
    print("test_color")
    x = torch.rand(2, 3, 16, 16)
    check("pq round-trip", torch.allclose(color.pq_eotf(color.pq_oetf(x)), x, atol=1e-3))
    check("709 round-trip", torch.allclose(color.bt709_eotf(color.bt709_oetf(x)), x, atol=1e-3))
    ycc = color.rgb_to_ycbcr2020(x)
    rgb = color.ycbcr2020_to_rgb(ycc)
    check("ycbcr round-trip", torch.allclose(rgb, x, atol=1e-4))


def test_rqs():
    print("test_rqs")
    b, k, n = 3, 8, 4096
    x = torch.rand(b, n)
    uw, uh = torch.randn(b, k), torch.randn(b, k)
    ud = torch.randn(b, k + 1)
    y = monotone_rqs(x, uw, uh, ud)
    check("rqs range [0,1]", (y.min() >= -1e-4) and (y.max() <= 1 + 1e-4))
    xs, _ = torch.sort(torch.rand(b, n), dim=1)
    ys = monotone_rqs(xs, uw, uh, ud)
    check("rqs monotone", (ys[:, 1:] - ys[:, :-1] >= -1e-4).all().item())
    dec = RQSToneDecoder(latent_dim=32)
    out = dec(torch.rand(2, 3, 16, 16), torch.randn(2, 32))
    check("rqs decoder shape", out.shape == (2, 3, 16, 16))
    check("rqs smoothness scalar", dec.spline_smoothness().ndim == 0)


def build_tiny_lumaflux():
    from diffusers import FluxTransformer2DModel, AutoencoderKL
    transformer = FluxTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(4, 6, 6),
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(32, 32, 32, 32), layers_per_block=1,
        scaling_factor=0.3611, shift_factor=0.1159,
    )
    cfg = LumaFluxConfig(mock_siglip=True, phys_dim=16, global_dim=16, num_bands=8, pga_rank=4)
    return LumaFlux(transformer, vae, cfg)


def test_wiring():
    print("test_wiring (tiny FLUX, CPU)")
    torch.manual_seed(0)
    model = build_tiny_lumaflux()
    n_train = model.num_trainable()
    check("has trainable params", n_train > 0)
    check("backbone frozen", all(not p.requires_grad for p in model.transformer.parameters()))
    print(f"    trainable params: {n_train/1e6:.3f}M")

    ds = SyntheticPairDataset(length=4, size=64)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    sdr, hdr = batch["sdr"], batch["hdr"]

    loss_fn = LumaFluxLoss(fm_weight=0.1)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)

    model.train()
    x_hat, v_pred, v_target = model.training_step(sdr, hdr)
    check("output shape", x_hat.shape == hdr.shape)
    check("output range [0,1]", (x_hat.min() >= -1e-4) and (x_hat.max() <= 1 + 1e-4))
    loss, logs = loss_fn(x_hat, hdr, model.rqs.spline_smoothness(), v_pred, v_target)
    check("loss finite", torch.isfinite(loss).item())
    loss.backward()

    grads = [p.grad for p in model.trainable_parameters() if p.grad is not None]
    check("grads populated", len(grads) > 0)
    check("grads finite", all(torch.isfinite(g).all().item() for g in grads))
    opt.step()
    print(f"    loss={loss.item():.4f}  logs=" +
          " ".join(f"{k}={v.item():.3f}" for k, v in logs.items()))

    # inference path
    model.eval()
    with torch.no_grad():
        hdr_pred = model.sample(sdr, num_steps=3)
    check("sample shape", hdr_pred.shape == sdr.shape)
    check("sample range [0,1]", (hdr_pred.min() >= -1e-4) and (hdr_pred.max() <= 1 + 1e-4))


if __name__ == "__main__":
    test_color()
    test_rqs()
    test_wiring()
    print("\nALL SMOKE TESTS PASSED")

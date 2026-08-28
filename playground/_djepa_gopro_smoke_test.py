import os
import socket
import time

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, "/home/wweligam/svd_v2")

print("=" * 70)
print(f"hostname: {socket.gethostname()}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
print(f"SLURM_JOB_PARTITION: {os.environ.get('SLURM_JOB_PARTITION')}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

from models.Baselines.DJEPA.djepa_wrapper import DJEPARestorer
from utils.misc import calculate_psnr, calculate_ssim

GOPRO_TEST = "/depot/chan129/data/GoPro/test"
OUT_DIR = "/scratch/gilbreth/wweligam/djepa_baseline_test/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_FRAME = [
    ("GOPR0868_11_00", "000013.png"),
    ("GOPR0871_11_00", "000037.png"),
    ("GOPR0862_11_00", "000059.png"),
    ("GOPR0410_11_00", "000070.png"),
]


def load_pm1(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    return t * 2 - 1, arr


def to01(t):
    return (t[0].detach().cpu().float().numpy() * 0.5 + 0.5).clip(0, 1).transpose(1, 2, 0)


t0 = time.time()
restorer = DJEPARestorer(variant="base", img_size=256, num_sampling_steps="25")
restorer.model.to("cuda")
restorer.vae.to("cuda")
print(f"Loaded D-JEPA-base in {time.time() - t0:.1f}s")

results = []
for seq, frame in SEQ_FRAME:
    blur_path = os.path.join(GOPRO_TEST, seq, "blur", frame)
    sharp_path = os.path.join(GOPRO_TEST, seq, "sharp", frame)
    if not (os.path.exists(blur_path) and os.path.exists(sharp_path)):
        print(f"SKIP missing {seq}/{frame}")
        continue

    blur_pm1, blur01 = load_pm1(blur_path)
    _, gt01 = load_pm1(sharp_path)

    row = {"seq": seq, "frame": frame}
    psnr_input = calculate_psnr(blur01, gt01, crop=30)
    ssim_input, _ = calculate_ssim(blur01, gt01, crop=30)
    row["psnr_blur_vs_gt"] = float(psnr_input)
    row["ssim_blur_vs_gt"] = float(ssim_input)

    for strength in [0.3, 0.6]:
        t0 = time.time()
        out = restorer.restore(blur_pm1, strength=strength, num_iter=64, cfg_scale=1.0)
        torch.cuda.synchronize()
        dt = time.time() - t0
        out01 = to01(out)

        psnr = calculate_psnr(out01, gt01, crop=30)
        ssim, _ = calculate_ssim(out01, gt01, crop=30)
        row[f"psnr_s{strength}"] = float(psnr)
        row[f"ssim_s{strength}"] = float(ssim)
        row[f"time_s{strength}"] = dt

        Image.fromarray((out01 * 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, f"{seq}_{frame}_s{strength}.png")
        )
        print(f"{seq}/{frame} strength={strength}: PSNR={psnr:.2f} SSIM={ssim:.3f} ({dt:.1f}s)")

    Image.fromarray((blur01 * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, f"{seq}_{frame}_blur.png"))
    Image.fromarray((gt01 * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, f"{seq}_{frame}_gt.png"))
    print(f"{seq}/{frame} blur-vs-gt (no restoration): PSNR={psnr_input:.2f} SSIM={ssim_input:.3f}")
    results.append(row)

print("=" * 70)
print("SUMMARY")
for r in results:
    print(r)
if results:
    import statistics
    print(f"mean PSNR blur-vs-gt: {statistics.mean(r['psnr_blur_vs_gt'] for r in results):.2f}")
    print(f"mean PSNR s=0.3:      {statistics.mean(r['psnr_s0.3'] for r in results):.2f}")
    print(f"mean PSNR s=0.6:      {statistics.mean(r['psnr_s0.6'] for r in results):.2f}")
print("DONE")

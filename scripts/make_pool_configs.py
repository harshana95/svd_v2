"""Generate the camera-shake + extreme-motion + FMO training-pool configs (Stage A and Stage B).

Pool (S=1, 512px, vjepa_window_size 8, wan_num_frames 9, wan_frames_range pm1 on EVERY source):
  Adobe240, GoPro (train scenes), X4K1000FPS (train scenes), OMoBlur (mask_crop), FMO/TbD (12 train scenes).
Held-out test sets, one per source (never trained on): Adobe240 test (8 scenes), GoPro test (11 scenes),
X4K scene_split [0.95,1.0], OMoBlur val, FMO/TbD (key, marker, out2, outf1).

Templates: Stage A = P2_train_ref (oracle V-JEPA tokens, cross-attention), Stage B = singleclip_win8_512.
Both are re-pointed to a FRESH run (no resume) so the 'trained only on the pool' claim holds.

    python scripts/make_pool_configs.py
"""
import copy
import os

import yaml

D = "checkpoints/MotionBlur2Video"
A_TEMPLATE = f"{D}/Wan22VJEPADynamics_P2_train_ref.yml"
B_TEMPLATE = f"{D}/BlurJepaViT_stageB_singleclip_win8_512.yml"

# train-mix weights (relative). FMO is small (12 scenes, memorises easily) -> kept low.
W = {"adobe": 0.25, "gopro": 0.20, "x4k": 0.20, "omoblur": 0.25, "fmo": 0.05, "tbd3d": 0.05}
X4K_DIR = "/scratch/gilbreth/wweligam/data/X4K1000FPS"
OMO_DIR = "/scratch/gilbreth/wweligam/OMoBlur"


def common(wan, stage):
    c = {"burst_size": 1, "vjepa_window_size": 8}
    if stage == "A":
        c.update(include_wan_frames=True, wan_num_frames=9, wan_key="wan_frames", wan_frames_range="pm1")
    else:
        c.update(include_wan_frames=False)
    return c


def sources(split, stage):
    """Source list for 'train' or 'val' (test sets)."""
    tr = split == "train"
    c = common(None, stage)
    adobe = dict(type="Adobe240Dataset", data_dir="/depot/chan129/data/Adobe240/Adobe_240fps_dataset/Adobe_240fps_blur",
                 split="train" if tr else "test", use_shuffle=tr, crop_size=352, out_size=512, window_size=11, **c)
    gopro = dict(type="GoProSequenceDataset", data_dir="/scratch/gilbreth/wweligam/data/GOPRO_Large_all",
                 split="train" if tr else "test", frames_subdir="train" if tr else "test", use_shuffle=tr,
                 random_crop=tr, crop_size=512, window_size=11, center_stride=8, **c)
    x4k = dict(type="X4K1000FPSDataset", data_dir=X4K_DIR, split="train" if tr else "val",
               is_train=tr, scene_split=[0.0, 0.95] if tr else [0.95, 1.0], use_shuffle=tr, random_crop=tr,
               crop_size=512, window_size=11, center_stride=8, **c)
    omo = dict(type="OMoBlurDataset", data_dir=OMO_DIR, split="train" if tr else "val", use_shuffle=tr,
               random_crop=tr, crop_size=360, out_size=512, blur_lengths=[11], **c)
    if tr:
        omo.update(mask_crop=True, mask_min_coverage=0.25, mask_blur_free_prob=0.1)
    fmo_common = dict(sharp_subdir="imgs_gt", out_size=512, window_size=11, center_stride=8, margin_frames=20, **c)
    fmo_dir = "/scratch/gilbreth/wweligam/data/FMO_TbD"
    if tr:
        fmo = dict(type="FallingObjectsDataset", data_dir=f"{fmo_dir}/falling_objects",
                   scenes=["v_box_GTgamma", "v_cell_GTgamma", "v_pen_GTgamma", "v_rubber_GTgamma"],
                   peak_center_scenes={"v_box_GTgamma": 301, "v_cell_GTgamma": 156, "v_pen_GTgamma": 341,
                                       "v_rubber_GTgamma": 304},
                   centers_around_peak=96, burst_center_step=1, crop_mode="random_square", **fmo_common)
        tbd = dict(type="FallingObjectsDataset", data_dir=f"{fmo_dir}/tbd3d",
                   scenes=["HighFPS_GT_depth2", "HighFPS_GT_depthb2", "HighFPS_GT_depthf1", "HighFPS_GT_depthf2",
                           "HighFPS_GT_depthf3", "HighFPS_GT_out1", "HighFPS_GT_outa1", "HighFPS_GT_outb1"],
                   peak_center_scenes={"HighFPS_GT_depth2": 32, "HighFPS_GT_depthb2": 35, "HighFPS_GT_depthf1": 319,
                                       "HighFPS_GT_depthf2": 48, "HighFPS_GT_depthf3": 53, "HighFPS_GT_out1": 102,
                                       "HighFPS_GT_outa1": 27, "HighFPS_GT_outb1": 147},
                   centers_around_peak=96, burst_center_step=1, crop_mode="random_square", **fmo_common)
    else:
        fmo = dict(type="FallingObjectsDataset", data_dir=f"{fmo_dir}/falling_objects",
                   scenes=["v_key_GTgamma", "v_marker_GTgamma"],
                   peak_center_scenes={"v_key_GTgamma": 270, "v_marker_GTgamma": 313},
                   crop_mode="motion_square", **fmo_common)
        tbd = dict(type="FallingObjectsDataset", data_dir=f"{fmo_dir}/tbd3d",
                   scenes=["HighFPS_GT_out2", "HighFPS_GT_outf1"],
                   peak_center_scenes={"HighFPS_GT_out2": 57, "HighFPS_GT_outf1": 150},
                   crop_mode="motion_square", **fmo_common)
    out = []
    for key, d in (("adobe", adobe), ("gopro", gopro), ("x4k", x4k), ("omoblur", omo), ("fmo", fmo), ("tbd3d", tbd)):
        out.append({"weight": W[key] if tr else 1.0 / 6, "name": key, **d})
    return out


def build(template, stage, key, steps, ckpt_every, val_every, extra=None):
    cfg = yaml.safe_load(open(template))
    cfg["experiment_key"] = key
    cfg["burst_size"] = 1
    for split in ("train", "val"):
        ds = cfg["datasets"][split]
        ds.update(burst_size=1, crop_size=512, vjepa_window_size=8, sources=sources(split, stage))
        if split == "val":
            ds["epoch_size"] = 12  # 2 per source -> every test set represented in the training-time val
    cfg["path"]["resume_from_path"] = None
    cfg["path"]["resume_from_checkpoint"] = None
    cfg["path"].pop("use_same_dir", None)
    t = cfg["train"]
    t.update(max_train_steps=steps, checkpointing_steps=ckpt_every, validation_steps=val_every)
    if extra:
        cfg.update(extra)
    return cfg


HEADER = """# GENERATED by scripts/make_pool_configs.py -- edit the script, not this file.
# Pool config ({stage}): Adobe240 + GoPro + X4K1000FPS + OMoBlur(mask_crop) + FMO/TbD, S=1, 512px, FRESH run.
# Train mix (relative): {w}.
# Per-source held-out test sets (val section = one entry each; final numbers come from the per-dataset
# manifest evals, NOT from training-time validation): Adobe240 test, GoPro test, X4K scene_split [0.95,1.0],
# OMoBlur val, FMO/TbD key/marker/out2/outf1.
# To resume a walltime-limited run: copy this file, set path.resume_from_path to the run dir,
# resume_from_checkpoint: latest, path.use_same_dir: true (resume_from_path is IGNORED without use_same_dir).
# max_train_steps is ABSOLUTE.
"""


def dump(cfg, path, stage):
    with open(path, "w") as f:
        f.write(HEADER.format(stage=stage, w=", ".join(f"{k} {v}" for k, v in W.items())))
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote", path)


def build_stageA_v2(key, *, multilayer=False, reblur=0.0, unfreeze=False, tshift=None, tokrob=False, pixel=0.0,
                    blur_everyframe=False, steps=100000, smoke=False):
    """Stage A v2 (fixed after the 2026-10-06 audit): pool + multi-res patchified batches (F=9 only, so the blur window == the
    clip's exposure as at eval) + isotropic scale/flip/time-reverse aug + LoRA r=32 on attn+FFN. Every new lever is an opt-in kwarg so
    arms differ by ONE thing: multilayer V-JEPA (4 depths), reblur (gt_mean target), unfreeze (fp32 masters, 0.1x LR, no wd issue),
    tshift (shifted t sampling), tokrob (token dropout+noise), pixel (center-frame L1+LPIPS), blur_everyframe (blur latent at every frame).
    FRESH run (LoRA rank / shapes change)."""
    cfg = build(A_TEMPLATE, "A", key, steps, 2000, 10000)
    cfg["disable_dynamics_injection"] = False
    cfg["lora_r"], cfg["lora_alpha"] = 32, 64
    cfg["lora_target_modules"] = ["to_q", "to_k", "to_v", "to_out", "ffn.net.0.proj", "ffn.net.2"]  # attn1+attn2+FFN
    if unfreeze:
        cfg["unfreeze_modules"] = ["patch_embedding", "proj_out", "condition_embedder", "norm2"]
        cfg["unfreeze_lr_scale"] = 0.1
    if multilayer:
        cfg["vjepa_layers"] = [5, 11, 17, 23]
    if reblur > 0:
        # gt_mean: mean of the clip's OWN F frames (exposure-consistent); 'blur' would compare to the full 11-frame blur
        cfg.update(reblur_weight=reblur, reblur_t_max=0.3, reblur_loss="l1", reblur_target="gt_mean", reblur_max_samples=1,
                   reblur_max_pixels=384 * 384)
    if pixel > 0:
        cfg.update(pixel_loss_weight=pixel, pixel_loss="l1+lpips", pixel_t_max=0.3, reblur_max_samples=1,
                   pixel_max_pixels=384 * 384)
    if tshift is not None:
        cfg.update(t_sampling="logit_normal", t_shift=float(tshift), t_logit_mean=0.0, t_logit_std=1.0)
        if reblur > 0 or pixel > 0:
            cfg.update(t_low_frac=0.25, t_low_max=0.3)   # keep the low-noise-only decoded losses firing under a high-noise-heavy schedule
    if tokrob:
        cfg.update(token_dropout=0.2, token_noise_std=0.3)
    if blur_everyframe:
        cfg["blur_every_frame_conditioning"] = True
    tr = cfg["datasets"]["train"]
    # patchified mode: sources emit LARGE natives, MixedDataset draws per-batch crop side / aug
    for src in tr["sources"]:
        if src["name"] in ("adobe", "gopro", "x4k"):
            src["crop_size"] = 704
            src.pop("out_size", None)
    tr.update(patch_aspect_ratios=True, is_train=True, batch_size=2, patches_per_group=2,
              crop_size=512, crop_size_choices=[384, 512, 640, 704], crop_aspect_min=0.5, crop_aspect_max=2.0,
              shape_divisor=32, frame_keys=["wan_frames"],
              augment={"hflip": 0.5, "vflip": 0.0, "time_reverse": 0.5, "scale_range": [0.75, 1.5]},
              epoch_size=4000, num_worker_per_gpu=4)
    cfg["train"]["batch_size"] = 2   # real per-step batch comes from datasets.train.batch_size in this mode
    if smoke:
        cfg["experiment_key"] += "_SMOKE"
        cfg["train"].update(max_train_steps=4, validation_steps=4, checkpointing_steps=4)
        tr["epoch_size"] = 16
        cfg["datasets"]["val"]["epoch_size"] = 6
        cfg["val"]["max_val_steps"] = 2
        cfg["val"]["num_inference_steps"] = 4
    return cfg


V2_ARMS = {  # name -> kwargs. Every arm adds ONE lever on top of v2_base (same data pipeline/LoRA), so effects are attributable.
    "v2_base": dict(),
    "v2_tshift": dict(tshift=5.0),
    "v2_reblur": dict(reblur=0.1),
    "v2_pixel": dict(pixel=0.1),
    "v2_tokrob": dict(tokrob=True),
    "v2_blurevery": dict(blur_everyframe=True),
    "v2_unfreeze": dict(unfreeze=True),
    "v2_ml": dict(multilayer=True),
    # combined long run: levers 1-6 that are safe w.r.t. the oracle->Stage-C gap (NO multilayer until a decode probe exists)
    "v2_full": dict(tshift=5.0, reblur=0.1, pixel=0.1, tokrob=True, blur_everyframe=True, unfreeze=True),
}


if __name__ == "__main__":
    import sys
    if "--v2" in sys.argv:
        for name, kw in V2_ARMS.items():
            steps = 100000 if name == "v2_full" else 30000   # tokens only engage after ~8k steps -> short arms are uninformative
            key = f"Wan22VJEPADynamics_stageA_{name}"
            cfg = build_stageA_v2(key, steps=steps, **kw)
            cfg["train"]["validation_steps"] = 5000 if steps == 30000 else 10000
            dump(cfg, f"{D}/{key}.yml", f"Stage A v2 ({name})")
            dump(build_stageA_v2(key, steps=steps, smoke=True, **kw), f"{D}/{key}_SMOKE.yml", f"Stage A v2 ({name}) SMOKE")
        sys.exit(0)
    a = build(A_TEMPLATE, "A", "Wan22VJEPADynamics_pool_stageA_S1", 30000, 1000, 5000)
    a["disable_dynamics_injection"] = False
    dump(a, f"{D}/Wan22VJEPADynamics_pool_stageA_S1.yml", "Stage A")
    b = build(B_TEMPLATE, "B", "BlurJepaViT_pool_stageB_S1", 20000, 500, 1000)
    dump(b, f"{D}/BlurJepaViT_pool_stageB_S1.yml", "Stage B")
    # smoke variants: tiny budgets, separate experiment keys
    sa = copy.deepcopy(a); sa["experiment_key"] += "_SMOKE"; sa["train"].update(max_train_steps=3, validation_steps=3, checkpointing_steps=3)
    sa["datasets"]["train"]["epoch_size"] = 12; dump(sa, f"{D}/Wan22VJEPADynamics_pool_stageA_S1_SMOKE.yml", "Stage A SMOKE")
    sb = copy.deepcopy(b); sb["experiment_key"] += "_SMOKE"; sb["train"].update(max_train_steps=3, validation_steps=3, checkpointing_steps=3)
    sb["datasets"]["train"]["epoch_size"] = 12; dump(sb, f"{D}/BlurJepaViT_pool_stageB_S1_SMOKE.yml", "Stage B SMOKE")

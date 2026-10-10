"""GoProSequenceDataset: GT-SEQUENCE loader for GOPRO_Large_all, synthesizing
blur by averaging contiguous sharp 240fps frames.

Motivation: the GoPro copy this repo normally uses (`GOPRO_Large`, at
/depot/chan129/data/GoPro) ships `{blur, blur_gamma, sharp}` — exactly ONE
sharp frame per blurred image — and `Deblur_MIOR` likewise ships `{blur, gt}`
pairs. Neither can supply the sharp frame *sequence* that blur->video /
motion-recovery work needs. `GOPRO_Large_all` is the separate official release
containing every 240fps sharp frame (34,874 frames, 33 videos, 1280x720, all
scenes verified contiguous), so blur can be synthesized at any window size and
the surrounding real frames remain available as the target sequence.

Layout at data_dir (/scratch/gilbreth/wweligam/data/GOPRO_Large_all):
    train/<scene>/000001.png ...      22 scenes
    test/<scene>/044001.png  ...      11 scenes

Frame indices are scene-relative and do NOT all start at 1 (e.g.
GOPR0384_11_05 runs 44001..44450), so the valid centre range is derived from
each scene's own observed min/max rather than assumed.

Subclasses X4K1000FPSDataset to reuse its config parsing, crop / aspect-ratio /
deferred-collate machinery, burst grouping and Adobe240-compatible output
contract (`blur`/`frames` + optional `wan_frames`, or `burst_blur`/
`burst_frames`). Only two things differ: sample discovery (scene dirs of PNGs
with per-scene index ranges, rather than fixed-length mp4 clips) and the frame
loading backend (`_decode01` reads PNGs instead of decoding video).

Blur is the plain mean over `window_size` frames in [0,1] — the Adobe240
convention this repo's Wan22VJEPADynamics line is trained on, NOT the
gamma-correct linear-light averaging in motionblur_dataset.py. Note this
differs from how GOPRO_Large's own shipped blur was made (7-13 frame windows
with CRF linearisation, plus a separate `blur_gamma` variant), so blur
synthesized here is NOT bit-comparable to the standard GoPro benchmark inputs —
it is comparable to Adobe240/X4K blur produced by this repo.
"""
import os
from itertools import zip_longest
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils import data as data

from dataset.x4k1000fps_dataset import X4K1000FPSDataset


class GoProSequenceDataset(X4K1000FPSDataset):
    def __init__(self, opt):
        data.Dataset.__init__(self)
        self._init_common(opt)

        # Official GoPro split dirs are literally "train"/"test"; `split` both
        # selects the directory and (via _init_common) sets is_train, which is
        # what random_crop defaults from.
        self.frames_subdir = opt.get("frames_subdir", self.split)
        extensions = tuple(opt.get("extensions", [".png", ".jpg", ".jpeg"]))

        root = os.path.join(self.data_dir, self.frames_subdir)
        if not os.path.isdir(root):
            raise ValueError(
                f"{root} does not exist -- expected GOPRO_Large_all extracted with "
                f"{self.frames_subdir}/<scene>/<frames>. Available: "
                f"{sorted(os.listdir(self.data_dir)) if os.path.isdir(self.data_dir) else 'n/a'}"
            )
        all_scenes = sorted(
            d for d in os.listdir(root)
            if not d.startswith(".") and os.path.isdir(os.path.join(root, d))
        )
        if not all_scenes:
            raise ValueError(f"No scene directories under {root}")

        # Optional further holdout *within* a split dir (e.g. carving a val set
        # out of the 22 train scenes). Whole scenes only -- frames inside one
        # scene share content, so a frame-level split would leak. Default uses
        # every scene in the chosen split dir.
        #
        # A val set carved this way still has split=="train", so pass
        # `is_train: false` alongside it (or `random_crop: false`) -- otherwise
        # random_crop defaults to True and validation stops being reproducible
        # across checkpoints.
        scene_split = opt.get("scene_split", None)
        if scene_split is not None:
            lo = int(float(scene_split[0]) * len(all_scenes))
            hi = int(float(scene_split[1]) * len(all_scenes))
            scenes = all_scenes[lo:hi]
            if not scenes:
                raise ValueError(
                    f"scene_split {list(scene_split)} selects no scenes out of {len(all_scenes)}"
                )
        else:
            scenes = all_scenes

        # The class is a generic "<root>/<scene>/<contiguous numbered frames>" tree loader, also used for Adobe240
        # (frames_subdir: full_sharp) and FMO / TbD (frames_subdir: imgs_gt). Scene filters:
        #   scenes:          explicit list (all must exist)
        #   scene_list_dir:  keep scenes named by <scene>_im_list.txt files in this dir (absolute or relative to
        #                    data_dir), e.g. Adobe240's train_list / test_list -- full_sharp holds BOTH splits, so the
        #                    list is what keeps test scenes out of training
        want = opt.get("scenes", None)
        if want is not None:
            want = [str(s) for s in want]
            missing = [s for s in want if s not in all_scenes]
            if missing:
                raise ValueError(f"scenes not found under {root}: {missing}")
            scenes = [s for s in scenes if s in set(want)]
        list_dir = opt.get("scene_list_dir", None)
        if list_dir:
            list_dir = list_dir if os.path.isabs(list_dir) else os.path.join(self.data_dir, list_dir)
            listed = {f[: -len("_im_list.txt")] for f in os.listdir(list_dir) if f.endswith("_im_list.txt")}
            if not listed:
                raise ValueError(f"no <scene>_im_list.txt files in {list_dir}")
            scenes = [s for s in scenes if s in listed]
        if not scenes:
            raise ValueError(f"scene filters left no scenes under {root}")

        self._init_span(opt)
        # Span starts can be concentrated around a per-scene event (FMO / TbD: the fall), whose absolute frame index
        # is span_centers[scene]; starts s keep |s + span_len/2 - center| <= span_center_radius.
        span_centers = opt.get("span_centers", None)
        span_center_radius = int(opt.get("span_center_radius", 96))

        per_scene_samples: List[List[Tuple[str, List[int]]]] = []
        self.scene_meta = {}
        for scene in scenes:
            scene_dir = os.path.join(root, scene)
            names = sorted(f for f in os.listdir(scene_dir) if f.lower().endswith(extensions))
            if not names:
                continue
            stem, ext = os.path.splitext(names[0])
            pad_width = len(stem)
            indices = sorted(int(os.path.splitext(n)[0]) for n in names)
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"Non-contiguous frame indices in {scene_dir}")
            self.scene_meta[scene_dir] = (ext, pad_width)

            if self.span_len:
                starts = list(range(indices[0], indices[-1] - self.span_len + 2, self.span_stride))
                if span_centers is not None:
                    c = span_centers.get(scene, None)          # .get only (DictAsMember)
                    if c is None:
                        raise ValueError(f"span_centers has no entry for scene {scene}")
                    starts = [s for s in starts if abs(s + self.span_len / 2 - float(c)) <= span_center_radius]
                if starts:
                    per_scene_samples.append([(scene_dir, [s]) for s in starts])
                continue

            lo_c = indices[0] + self.half_window
            hi_c = indices[-1] - self.half_window
            centers_all = list(range(lo_c, hi_c + 1, self.center_stride))
            if len(centers_all) < self.burst_size:
                continue
            if self.burst_size == 1:
                per_scene_samples.append([(scene_dir, [c]) for c in centers_all])
            else:
                per_scene_samples.append([
                    (scene_dir, centers_all[s:s + self.burst_size])
                    for s in range(0, len(centers_all) - self.burst_size + 1, self.burst_size)
                ])

        # Round-robin across scenes so unshuffled iteration (this repo validates
        # with use_shuffle: false) spans many scenes rather than exhausting the
        # first one.
        self.samples: List[Tuple[str, List[int]]] = [
            s for group in zip_longest(*per_scene_samples) for s in group if s is not None
        ]
        if not self.samples:
            raise ValueError(f"No valid samples constructed from {root}")

        print(
            f"Initialized GoProSequenceDataset ({self.frames_subdir}) with {len(self.samples)} "
            f"samples across {len(per_scene_samples)} scenes "
            + (f"(span mode: span_len={self.span_len}, span_stride={self.span_stride})" if self.span_len else
               f"(window_size={self.window_size}, vjepa_window_size={self.vjepa_window_size}, "
               f"burst_size={self.burst_size}, center_stride={self.center_stride})")
        )

    def _span_name(self, scene_dir: str, start: int) -> str:
        return f"{self.frames_subdir}_{os.path.basename(scene_dir)}_f{start:06d}_L{self.span_len}"

    def _decode01(self, scene_dir: str, start: int, count: int) -> np.ndarray:
        """[count,H,W,3] float32 in [0,1], read from this scene's PNGs. Replaces
        the parent's decord video decode; `start` is an absolute scene-relative
        frame index, which is why the per-scene zero-pad width is recorded at
        discovery time rather than assumed."""
        ext, pad_width = self.scene_meta[scene_dir]
        return np.stack([
            np.asarray(
                Image.open(os.path.join(scene_dir, f"{i:0{pad_width}d}{ext}")).convert("RGB"),
                dtype=np.float32,
            ) / 255.0
            for i in range(start, start + count)
        ])

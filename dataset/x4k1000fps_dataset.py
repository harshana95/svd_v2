"""X4K1000FPSDataset: drop-in replacement for Adobe240Dataset backed by
X4K1000FPS (XVFI, ICCV 2021) instead of Adobe240fps, decoding the source mp4
clips ON THE FLY rather than pre-expanding them to PNG.

Layout at data_dir (e.g. /scratch/gilbreth/wweligam/data/X4K1000FPS):
    encoded_train/<scene>/<clip>.mp4    4,408 clips, 65 frames each, 768x768

Emits exactly the same sample contract as Adobe240Dataset -- `blur`/`frames`
(+ optional `wan_frames`), or `burst_blur`/`burst_frames` when burst_size > 1 --
with the same crop/aspect-ratio/collate machinery, so an existing
Wan22VJEPADynamics_* config swaps over by changing only `type` and `data_dir`.

Why on-the-fly decode: the official `mp4_decoding.py` expands X-TRAIN to ~240 GB
and ~286K PNGs. Decoding the needed span per __getitem__ instead keeps the
dataset at the 14 GB of extracted mp4 already on disk. Clips are ~1.5 MB, so a
per-call open is cheap; only the contiguous span a sample needs is decoded.

Blur synthesis: plain mean over `window_size` frames in [0,1], matching the
construction Adobe240's own pre-synthesized blur files were verified to use
(see adobe240_dataset.py's docstring). Deliberately NOT the gamma-correct
linear-light averaging in motionblur_dataset.py's build_blur() -- that is the
blur2vid branch's convention, and mixing the two would confound any comparison
against Adobe240-trained checkpoints.

Timing note: X4K frames are 1 ms apart vs Adobe240's 4.17 ms, so an identical
`window_size` spans ~4x less real time. It does NOT produce ~4x less blur:
X4K is curated for extreme motion (it is the "eXtreme Video Frame
Interpolation" benchmark), and measured per-frame displacement is comparable to
Adobe240's. At window_size=11 the synthesized blur is actually *stronger* here
(29.0 dB blur-vs-center-frame) than on Adobe240 (32.7 dB), so the default
window carries over directly rather than needing to be scaled up. The 65-frame
clip length caps the reachable window regardless.
"""
import math
import os
import random
from itertools import zip_longest
from typing import List, Tuple

import cv2
import decord
import numpy as np
from PIL import Image
from torch.utils import data as data
from torch.utils.data import default_collate


class X4K1000FPSDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self._init_common(opt)
        self._init_span(opt)
        self.clips_subdir = opt.get("clips_subdir", "encoded_train")
        self._discover(opt)

    # ------------------------------------------------------------------ span mode
    def _init_span(self, opt):
        """Span mode (stroke-controlled Stage B, BlurJepaStrokeControl_model): each sample is a contiguous run of
        `span_len` raw sharp frames and the MODEL cuts its blur exposures out of it (variable exposure length,
        contiguous segments), so no blur / window / burst grouping happens here. Off by default (span_len unset).

        span_geometry:
          crop    (default; the phase-1 behaviour) resize_mode: resize -> _resize01, else a crop_size square
          full    the whole frame, aspect kept, downscaled to a long side <= span_max_side
          random  per sample: `full` with probability span_p_full, else a crop with aspect (w/h) log-uniform in
                  span_crop_aspect and area fraction uniform in span_crop_area, then downscaled like `full`
        Non-square / variable-size samples need batch_size 1 (the stroke model letterboxes every sample onto V-JEPA's
        square canvas itself)."""
        self.span_len = int(opt.get("span_len", 0) or 0) or None
        self.span_stride = int(opt.get("span_stride", self.span_len or 1))
        self.span_key = opt.get("span_key", "span_frames")
        self.span_geometry = str(opt.get("span_geometry", "crop"))
        if self.span_geometry not in ("crop", "full", "random"):
            raise ValueError(f"Unknown span_geometry: {self.span_geometry!r}")
        self.span_p_full = float(opt.get("span_p_full", 0.5))
        self.span_crop_aspect = tuple(float(x) for x in opt.get("span_crop_aspect", [0.5, 2.0]))
        self.span_crop_area = tuple(float(x) for x in opt.get("span_crop_area", [0.2, 1.0]))
        self.span_max_side = int(opt.get("span_max_side", 768))
        if self.span_len and self.aspect_ratio_range is not None:
            raise ValueError("span mode does not use aspect_ratio_range (use span_geometry: random)")

    def _span_downscale(self, frames01: np.ndarray) -> np.ndarray:
        h, w = frames01.shape[1:3]
        s = self.span_max_side / max(h, w)
        if s >= 1.0:
            return frames01
        nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
        return np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames01])

    def _span_apply_geometry(self, frames01: np.ndarray) -> np.ndarray:
        if self.span_geometry == "crop":
            if self.resize_mode == "resize":
                return self._resize01(frames01)
            h, w = frames01.shape[1:3]
            ch, cw = min(self.crop_size, h), min(self.crop_size, w)
            top, left = self._crop_region(h, w, ch, cw)
            return frames01[:, top:top + ch, left:left + cw]
        if self.span_geometry == "random" and random.random() >= self.span_p_full:
            h, w = frames01.shape[1:3]
            lo, hi = self.span_crop_aspect
            aspect = math.exp(random.uniform(math.log(lo), math.log(hi)))          # w / h
            area = random.uniform(*self.span_crop_area) * h * w
            ch = math.sqrt(area / aspect)
            cw = ch * aspect
            if cw > w:
                cw, ch = w, min(h, area / w)
            if ch > h:
                ch, cw = h, min(w, area / h)
            ch, cw = max(16, int(ch)), max(16, int(cw))
            top, left = random.randint(0, h - ch), random.randint(0, w - cw)
            frames01 = frames01[:, top:top + ch, left:left + cw]
        return self._span_downscale(frames01)

    def _span_name(self, src: str, start: int) -> str:
        scene = os.path.basename(os.path.dirname(src))
        clip = os.path.splitext(os.path.basename(src))[0].replace(".", "_")
        return f"{scene}_{clip}_f{start:03d}_L{self.span_len}"

    def _span_item(self, idx: int) -> dict:
        src, (start,) = self.samples[idx]
        frames01 = self._span_apply_geometry(self._decode01(src, start, self.span_len))   # [L,H,W,3] in [0,1]
        sample = {self.span_key: np.ascontiguousarray(frames01.transpose(0, 3, 1, 2), dtype=np.float32)}
        if self.return_name:
            sample["name"] = self._span_name(src, start)
        return sample

    def _init_common(self, opt):
        """Config shared with any contiguous-sharp-frames source (see
        GoProSequenceDataset, which subclasses this) -- everything except
        sample discovery and the frame-loading backend."""
        self.data_dir = opt.data_dir
        self.split = opt.get("split", "train")
        # Overridable because `split` doubles as a directory selector in
        # GoProSequenceDataset: a val set carved out of the train/ dir via
        # scene_split still has split=="train", which would otherwise default
        # random_crop to True and make validation non-reproducible across
        # checkpoints.
        self.is_train = bool(opt.get("is_train", self.split == "train"))

        self.gt_key = opt.get("gt_key", "frames")
        self.lq_key = opt.get("lq_key", "blur")
        self.return_name = opt.get("return_name", True)

        self.crop_size = int(opt.get("crop_size", 384))
        self.random_crop = bool(opt.get("random_crop", self.is_train))

        ar_range = opt.get("aspect_ratio_range", None)
        self.aspect_ratio_range = tuple(ar_range) if ar_range is not None else None
        if self.aspect_ratio_range is not None and len(self.aspect_ratio_range) != 2:
            raise ValueError(f"aspect_ratio_range must be [lo, hi], got {self.aspect_ratio_range!r}")
        self.crop_align = int(opt.get("crop_align", 32))

        self.resize_mode = opt.get("resize_mode", "crop")
        if self.resize_mode not in ("crop", "resize"):
            raise ValueError(f"Unknown resize_mode: {self.resize_mode!r}")
        self.resize_h = int(opt.get("resize_h", 720))
        self.resize_w = int(opt.get("resize_w", 1280))

        self.window_size = int(opt.get("window_size", 11))
        self.vjepa_window_size = int(opt.get("vjepa_window_size", 10))
        if self.vjepa_window_size > self.window_size:
            raise ValueError("vjepa_window_size cannot exceed window_size")
        self.half_window = self.window_size // 2
        # blur_window: how many CENTER frames of the loaded window the blur averages (default = the whole window = old behaviour)
        self.blur_window = int(opt.get("blur_window", self.window_size))
        if self.blur_window > self.window_size or (self.window_size - self.blur_window) % 2:
            raise ValueError("blur_window must be <= window_size with the same parity")
        self.blur_off = (self.window_size - self.blur_window) // 2
        # v3 context frames: when `ctx_key` is set the WHOLE loaded real window (window_size frames, e.g. 19) is also emitted under
        # that key, [window_size,3,H,W] in the wan_frames_range; wan_frames stays the centered `wan_num_frames` exposure slice.
        self.ctx_key = opt.get("ctx_key", None)

        # Adobe240's listed blur centers are 8 frames apart; kept as the
        # default so burst spans line up with what the model was trained on.
        self.center_stride = int(opt.get("center_stride", 8))

        self.include_wan_frames = bool(opt.get("include_wan_frames", False))
        self.wan_num_frames = int(opt.get("wan_num_frames", 9))
        self.wan_vae_scale_factor_temporal = int(opt.get("wan_vae_scale_factor_temporal", 4))
        self.wan_prefix_frames = int(opt.get("wan_prefix_frames", 0))
        self.wan_key = opt.get("wan_key", "wan_frames")
        self.wan_frames_range = opt.get("wan_frames_range", "pm1")

        self.burst_size = int(opt.get("burst_size", 1))
        if self.burst_size < 1:
            raise ValueError(f"burst_size must be >= 1, got {self.burst_size}")
        self.burst_key = opt.get("burst_key", "burst_blur")
        self.burst_frames_key = opt.get("burst_frames_key", "burst_frames")

        if self.include_wan_frames:
            if self.burst_size == 1 and self.wan_num_frames > self.window_size:
                raise ValueError("wan_num_frames cannot exceed window_size")
            if (self.wan_num_frames + self.wan_prefix_frames - 1) % self.wan_vae_scale_factor_temporal != 0:
                raise ValueError(
                    f"wan_num_frames={self.wan_num_frames} must satisfy "
                    f"(wan_num_frames + wan_prefix_frames - 1) % wan_vae_scale_factor_temporal == 0 "
                    f"(wan_vae_scale_factor_temporal={self.wan_vae_scale_factor_temporal})"
                )
            if self.wan_frames_range not in ("pm1", "01"):
                raise ValueError(f"Unknown wan_frames_range: {self.wan_frames_range!r}")

    def _discover(self, opt):
        clips_root = os.path.join(self.data_dir, self.clips_subdir)
        if not os.path.isdir(clips_root):
            raise ValueError(
                f"{clips_root} does not exist -- extract encoded_train.tar.gz first "
                f"(`tar -xzf encoded_train.tar.gz` in {self.data_dir})."
            )
        # The shipped tarball contains a stray `.ipynb_checkpoints` dir alongside
        # the 110 real scenes; unfiltered it sorts first (leading dot) and shifts
        # every scene_split boundary by one while contributing no clips.
        all_scenes = sorted(
            d for d in os.listdir(clips_root)
            if not d.startswith(".") and os.path.isdir(os.path.join(clips_root, d))
        )
        if not all_scenes:
            raise ValueError(f"No scene directories under {clips_root}")

        # X4K ships no train/val split, so hold out WHOLE SCENES by fraction of
        # the sorted scene list -- splitting by clip would leak, since clips
        # from one scene share content. Default gives val the last 5%.
        default_scene_split = [0.0, 0.95] if self.is_train else [0.95, 1.0]
        scene_split = opt.get("scene_split", default_scene_split)
        lo = int(float(scene_split[0]) * len(all_scenes))
        hi = int(float(scene_split[1]) * len(all_scenes))
        scenes = all_scenes[lo:hi]
        if not scenes:
            raise ValueError(
                f"scene_split {list(scene_split)} selects no scenes out of {len(all_scenes)}"
            )

        # Clip length is uniform (65) across X-TRAIN; probe one clip rather
        # than trusting the paper, but don't open all 4,408 at init.
        self.clip_length = opt.get("clip_length", None)
        probe_clip = None
        for scene in scenes:
            names = sorted(
                f for f in os.listdir(os.path.join(clips_root, scene)) if f.endswith(".mp4")
            )
            if names:
                probe_clip = os.path.join(clips_root, scene, names[0])
                break
        if probe_clip is None:
            raise ValueError(f"No .mp4 clips found under {clips_root} for the selected scenes")
        probed = len(decord.VideoReader(probe_clip, num_threads=1))
        if self.clip_length is None:
            self.clip_length = int(probed)
        elif int(self.clip_length) != probed:
            raise ValueError(
                f"clip_length={self.clip_length} but {probe_clip} has {probed} frames"
            )
        self.clip_length = int(self.clip_length)

        # Centers whose full window_size span fits inside the clip.
        first_center = self.half_window
        last_center = self.clip_length - 1 - self.half_window
        centers_all = list(range(first_center, last_center + 1, self.center_stride))
        if len(centers_all) < self.burst_size:
            raise ValueError(
                f"clip_length={self.clip_length} with window_size={self.window_size} and "
                f"center_stride={self.center_stride} yields only {len(centers_all)} centers, "
                f"fewer than burst_size={self.burst_size}"
            )

        # One entry per clip, mirroring Adobe240Dataset's per-scene grouping:
        # bursts are non-overlapping chunks of burst_size consecutive centers.
        per_clip_samples: List[List] = []
        n_clips = 0
        for scene in scenes:
            scene_dir = os.path.join(clips_root, scene)
            for name in sorted(f for f in os.listdir(scene_dir) if f.endswith(".mp4")):
                clip_path = os.path.join(scene_dir, name)
                n_clips += 1
                if self.span_len:
                    starts = list(range(0, self.clip_length - self.span_len + 1, self.span_stride))
                    if starts:
                        per_clip_samples.append([(clip_path, [s]) for s in starts])
                elif self.burst_size == 1:
                    per_clip_samples.append([(clip_path, [c]) for c in centers_all])
                else:
                    per_clip_samples.append([
                        (clip_path, centers_all[s:s + self.burst_size])
                        for s in range(0, len(centers_all) - self.burst_size + 1, self.burst_size)
                    ])

        # Interleave round-robin across clips so unshuffled iteration (this
        # repo validates with use_shuffle: false) spans many clips instead of
        # drawing every validation sample from whichever clip sorts first.
        self.samples: List[Tuple[str, List[int]]] = [
            s for group in zip_longest(*per_clip_samples) for s in group if s is not None
        ]
        if not self.samples:
            raise ValueError(f"No valid samples constructed from {clips_root}")

        print(
            f"Initialized X4K1000FPSDataset ({self.split}) with {len(self.samples)} samples "
            f"across {n_clips} clips / {len(scenes)} scenes (window_size={self.window_size}, "
            f"vjepa_window_size={self.vjepa_window_size}, burst_size={self.burst_size}, "
            f"center_stride={self.center_stride}, clip_length={self.clip_length})"
        )

    def __len__(self):
        return len(self.samples)

    def _decode01(self, clip_path: str, start: int, count: int) -> np.ndarray:
        """[count,H,W,3] float32 in [0,1], decoded from `clip_path`.

        The VideoReader is created here rather than cached on self: a reader
        built in the parent process and then used inside forked DataLoader
        workers is the same fork-after-init hazard that produced this repo's
        full-res eval deadlock. Clips are ~1.5 MB, so re-opening is cheap.
        """
        vr = decord.VideoReader(clip_path, num_threads=1)
        frames = vr.get_batch(list(range(start, start + count))).asnumpy()
        return frames.astype(np.float32) / 255.0

    def _resize01(self, frames01: np.ndarray) -> np.ndarray:
        """[T,H,W,3] -> [T,resize_h,resize_w,3], non-aspect-preserving, matching
        Adobe240Dataset._load01_resized's PIL convention."""
        out = []
        for f in frames01:
            img = Image.fromarray((f * 255.0).astype(np.uint8)).resize(
                (self.resize_w, self.resize_h)
            )
            out.append(np.asarray(img, dtype=np.float32) / 255.0)
        return np.stack(out)

    def _sample_crop_hw(self, native_h: int, native_w: int) -> Tuple[int, int]:
        if not (self.random_crop and self.aspect_ratio_range is not None):
            return self.crop_size, self.crop_size
        lo, hi = self.aspect_ratio_range
        ar = random.uniform(lo, hi)
        target_area = float(self.crop_size * self.crop_size)
        h = math.sqrt(target_area / ar)
        w = target_area / h
        if h > native_h:
            h = float(native_h)
            w = target_area / h
        if w > native_w:
            w = float(native_w)
            h = target_area / w
        h = min(h, native_h)
        w = min(w, native_w)
        crop_h = max(self.crop_align, int(h) // self.crop_align * self.crop_align)
        crop_w = max(self.crop_align, int(w) // self.crop_align * self.crop_align)
        return crop_h, crop_w

    def _crop_region(self, h: int, w: int, crop_h: int, crop_w: int) -> Tuple[int, int]:
        if h < crop_h or w < crop_w:
            return 0, 0
        if self.random_crop:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
        else:
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
        return top, left

    def _defer_crop_to_collate(self) -> bool:
        return self.resize_mode == "crop" and self.random_crop and self.aspect_ratio_range is not None

    def _finalize_sample(self, blur01: np.ndarray, frames01_full: np.ndarray, name: str) -> dict:
        blur_pm1 = (blur01.transpose(2, 0, 1) * 2.0 - 1.0).astype(np.float32)
        sample = {self.lq_key: blur_pm1}

        if self.include_wan_frames:
            wan_start = (self.window_size - self.wan_num_frames) // 2
            wan_frames01 = frames01_full[wan_start:wan_start + self.wan_num_frames]
            wan_frames01 = wan_frames01.transpose(0, 3, 1, 2).astype(np.float32)
            if self.wan_frames_range == "pm1":
                wan_frames01 = wan_frames01 * 2.0 - 1.0
            sample[self.wan_key] = wan_frames01
            if self.ctx_key:
                ctx01 = frames01_full.transpose(0, 3, 1, 2).astype(np.float32)
                sample[self.ctx_key] = ctx01 * 2.0 - 1.0 if self.wan_frames_range == "pm1" else ctx01

        frames01 = frames01_full[: self.vjepa_window_size].transpose(0, 3, 1, 2).astype(np.float32)
        sample[self.gt_key] = frames01

        if self.return_name:
            sample["name"] = name
        return sample

    def _finalize_burst_sample(
        self,
        burst_blur01: np.ndarray,
        full_frames01: np.ndarray,
        centers: List[int],
        full_span_start: int,
        name: str,
    ) -> dict:
        burst_blur_pm1 = (burst_blur01.transpose(0, 3, 1, 2) * 2.0 - 1.0).astype(np.float32)
        sample = {self.burst_key: burst_blur_pm1}

        if self.include_wan_frames:
            full_span_len = full_frames01.shape[0]
            wan_start = (full_span_len - self.wan_num_frames) // 2
            if wan_start < 0:
                raise ValueError(
                    f"wan_num_frames={self.wan_num_frames} exceeds the full burst span "
                    f"({full_span_len} frames, burst_size={self.burst_size}); reduce "
                    f"wan_num_frames or increase burst_size/center_stride."
                )
            wan_frames01 = full_frames01[wan_start:wan_start + self.wan_num_frames]
            wan_frames01 = wan_frames01.transpose(0, 3, 1, 2).astype(np.float32)
            if self.wan_frames_range == "pm1":
                wan_frames01 = wan_frames01 * 2.0 - 1.0
            sample[self.wan_key] = wan_frames01

        burst_frames = []
        for c in centers:
            seg_start = (c - self.half_window) - full_span_start
            seg_window = full_frames01[seg_start:seg_start + self.window_size]
            burst_frames.append(seg_window[: self.vjepa_window_size])
        burst_frames = np.stack(burst_frames).transpose(0, 1, 4, 2, 3).astype(np.float32)
        sample[self.burst_frames_key] = burst_frames

        if self.return_name:
            sample["name"] = name
        return sample

    def collate_fn(self, batch: List[dict]):
        if not batch:
            return default_collate(batch)

        if "_native_burst_blur01" in batch[0]:
            native_h = min(item["_native_burst_blur01"].shape[1] for item in batch)
            native_w = min(item["_native_burst_blur01"].shape[2] for item in batch)
            crop_h, crop_w = self._sample_crop_hw(native_h, native_w)
            finalized = []
            for item in batch:
                burst_blur01_native = item["_native_burst_blur01"]
                full_frames01_native = item["_native_burst_full_frames01"]
                h, w = burst_blur01_native.shape[1:3]
                ch, cw = min(crop_h, h), min(crop_w, w)
                top, left = self._crop_region(h, w, ch, cw)
                finalized.append(self._finalize_burst_sample(
                    burst_blur01_native[:, top:top + ch, left:left + cw],
                    full_frames01_native[:, top:top + ch, left:left + cw],
                    item["_burst_centers"], item["_burst_full_span_start"], item["_name"],
                ))
            return default_collate(finalized)

        if "_native_blur01" not in batch[0]:
            return default_collate(batch)
        native_h = min(item["_native_blur01"].shape[0] for item in batch)
        native_w = min(item["_native_blur01"].shape[1] for item in batch)
        crop_h, crop_w = self._sample_crop_hw(native_h, native_w)
        finalized = []
        for item in batch:
            blur01_native = item["_native_blur01"]
            frames01_full_native = item["_native_frames01"]
            h, w = blur01_native.shape[:2]
            ch, cw = min(crop_h, h), min(crop_w, w)
            top, left = self._crop_region(h, w, ch, cw)
            finalized.append(self._finalize_sample(
                blur01_native[top:top + ch, left:left + cw],
                frames01_full_native[:, top:top + ch, left:left + cw],
                item["_name"],
            ))
        return default_collate(finalized)

    def __getitem__(self, idx):
        if getattr(self, "span_len", None):
            return self._span_item(idx)
        clip_path, centers = self.samples[idx]
        full_span_start = centers[0] - self.half_window
        span_len = (centers[-1] + self.half_window) - full_span_start + 1

        full_frames01 = self._decode01(clip_path, full_span_start, span_len)
        if self.resize_mode == "resize":
            full_frames01 = self._resize01(full_frames01)

        # Each segment's blur is the plain mean of its own window_size frames,
        # sliced out of the single decoded span (consecutive segments' windows
        # overlap whenever center_stride < window_size, exactly as in Adobe240).
        burst_blur01 = np.stack([
            full_frames01[(c - self.half_window) - full_span_start + self.blur_off:
                          (c - self.half_window) - full_span_start + self.blur_off + self.blur_window].mean(axis=0)
            for c in centers
        ])

        scene = os.path.basename(os.path.dirname(clip_path))
        # X4K clip stems embed a decimal point (e.g. "occ008.367_f2129"), and
        # consumers of sample["name"] run os.path.splitext on it to build output
        # filenames (Wan22VJEPADynamics_model.py) -- left as-is that truncates
        # "002_occ008.367_f2129_c005" to "002_occ008", colliding across every
        # clip in the scene that shares a prefix. Strip the dots here.
        clip = os.path.splitext(os.path.basename(clip_path))[0].replace(".", "_")
        suffix = "" if self.burst_size == 1 else f"_burst{len(centers)}"
        name = f"{scene}_{clip}_c{centers[0]:03d}{suffix}"

        if self.resize_mode == "crop":
            h, w = full_frames01.shape[1:3]
            if self._defer_crop_to_collate():
                if self.burst_size == 1:
                    return {
                        "_native_blur01": burst_blur01[0],
                        "_native_frames01": full_frames01,
                        "_name": name,
                    }
                return {
                    "_native_burst_blur01": burst_blur01,
                    "_native_burst_full_frames01": full_frames01,
                    "_burst_centers": centers,
                    "_burst_full_span_start": full_span_start,
                    "_name": name,
                }
            crop_h, crop_w = self._sample_crop_hw(h, w)
            ch, cw = min(crop_h, h), min(crop_w, w)
            top, left = self._crop_region(h, w, ch, cw)
            burst_blur01 = burst_blur01[:, top:top + ch, left:left + cw]
            full_frames01 = full_frames01[:, top:top + ch, left:left + cw]

        if self.burst_size == 1:
            return self._finalize_sample(burst_blur01[0], full_frames01, name)
        return self._finalize_burst_sample(
            burst_blur01, full_frames01, centers, full_span_start, name
        )

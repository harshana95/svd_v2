"""Adobe240Dataset: pairs a pre-synthesized blurred frame (Adobe240fps,
window-averaged) with the real sharp high-frame-rate frames that produced it,
for training an encoder to predict V-JEPA tokens (computed from the real
frames) from the single blurred image alone. See
findings/ (or the approved plan) for the full design rationale.

Layout at data_dir (e.g. /depot/chan129/data/Adobe240/Adobe_240fps_dataset/Adobe_240fps_blur):
    full_sharp/<scene>/00001.png, 00002.png, ...   every extracted 240fps frame
    <split>/<scene>/*.png                           sharp frames for this split (unused here)
    <split>_blur/<scene>/*.png                       pre-synthesized blur frames
    <split>_list/<scene>_im_list.txt                 one filename per blur frame -- the
                                                      CENTER frame of the averaging window,
                                                      not an exhaustive frame list

Verified directly (see plan's verification step 1): the blur images were
synthesized by averaging an 11-frame window centered on the listed filename
(mean abs error minimized sharply at window=11 vs 7/9/13/15 -- 0.45 vs.
1.3-2.5 on a 0-255 scale, i.e. near-exact reconstruction). V-JEPA's loaded
hub model has tubelet_size=2 (verified via load_vjepa2_model(...).tubelet_size),
so an odd 11-frame clip gets silently truncated to 10 by the model itself --
rather than depend on that undocumented behavior, this dataset returns
`window_size` (default 11, used only to reconstruct/verify the blur image)
and separately exposes `vjepa_window_size` (default 10, an even prefix of
the same window) as the frame set actually meant to be fed to V-JEPA.
"""
import math
import os
import random
from itertools import zip_longest
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils import data as data
from torch.utils.data import default_collate


class Adobe240Dataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.data_dir = opt.data_dir
        self.split = opt.get("split", "train")
        self.is_train = self.split == "train"

        self.gt_key = opt.get("gt_key", "frames")
        self.lq_key = opt.get("lq_key", "blur")
        self.return_name = opt.get("return_name", True)

        self.crop_size = int(opt.get("crop_size", 384))
        self.random_crop = bool(opt.get("random_crop", self.is_train))

        # out_size: resize the FINAL cropped sample to (out_size, out_size).
        # None (default) = off, preserving all previous behavior.
        #
        # WHY this is needed separately from crop_size: Adobe240 frames are
        # natively 640x352, so a crop LARGER than 352 cannot be taken --
        # _crop_region returns (0,0) when the frame is smaller than the crop
        # and the slice then silently yields a short array, which only
        # surfaces later as a collate shape error. To reach a 512x512 canvas
        # this source must crop 352x352 at native pixel scale (full height,
        # random horizontal position, aspect preserved) and then upsample
        # 1.45x. Cropping and output size are genuinely different quantities
        # here; conflating them is what limited this source to 320.
        out_size = opt.get("out_size", None)
        self.out_size = int(out_size) if out_size is not None else None

        # Optional: vary the crop's aspect ratio (crop_w/crop_h) per training
        # batch -- kept consistent *within* a batch (via collate_fn, since
        # torch's default collate needs uniform tensor shapes to stack), but
        # varies *between* batches -- while holding area ~= crop_size^2
        # fixed, so the model sees genuinely different (H,W) token-grid
        # shapes during training instead of always a fixed square, to help
        # it generalize to different aspect ratios at test/inference time
        # (e.g. this repo's own hires-eval resolutions, or blur2vid's
        # 1280x720, vs. a 320x320 square training default). Only active
        # when random_crop is True -- validation (random_crop=False by
        # default) always uses the original deterministic center-square
        # crop regardless of this setting, for reproducible validation
        # across checkpoints (each validation "batch" is batch_size=1 in
        # every config here anyway, so there's no batching conflict to
        # avoid there in the first place).
        ar_range = opt.get("aspect_ratio_range", None)
        self.aspect_ratio_range = tuple(ar_range) if ar_range is not None else None
        if self.aspect_ratio_range is not None and len(self.aspect_ratio_range) != 2:
            raise ValueError(f"aspect_ratio_range must be [lo, hi], got {self.aspect_ratio_range!r}")
        # Both crop dims must be evenly divisible by
        # vae_scale_factor_spatial(16)*patch_size(2)=32 for Wan's
        # patch_embedding to round-trip losslessly (see
        # findings/Wan22VJEPADynamics_findings.md) -- 32 is that model's own
        # constraint, not a generic dataset default, so it's only ever
        # actually exercised when aspect_ratio_range is set.
        self.crop_align = int(opt.get("crop_align", 32))

        # resize_mode="resize": whole-frame, non-aspect-preserving resize to
        # (resize_h, resize_w) instead of a square crop -- matches
        # models/Baselines/Blur2Vid/blur2vid_lib/inference.py's own
        # convert_to_batch() convention (PIL .resize(), no cropping), so a
        # baseline evaluated at that resolution and our own model can be
        # compared at the same, consistent resolution. Additive: default
        # "crop" preserves all existing behavior.
        self.resize_mode = opt.get("resize_mode", "crop")
        if self.resize_mode not in ("crop", "resize"):
            raise ValueError(f"Unknown resize_mode: {self.resize_mode!r}")
        self.resize_h = int(opt.get("resize_h", 720))
        self.resize_w = int(opt.get("resize_w", 1280))

        # window_size: the true blur-synthesis window (verified = 11).
        # vjepa_window_size: an even prefix of that window actually handed to
        # V-JEPA (tubelet_size=2 needs an even frame count) -- kept separate
        # so the blur-image reconstruction check always uses the true,
        # verified window regardless of what's fed to V-JEPA.
        self.window_size = int(opt.get("window_size", 11))
        self.vjepa_window_size = int(opt.get("vjepa_window_size", 10))
        if self.vjepa_window_size > self.window_size:
            raise ValueError("vjepa_window_size cannot exceed window_size")
        self.half_window = self.window_size // 2
        # v3 context frames: when `ctx_key` is set the WHOLE loaded real window (window_size frames, e.g. 19) is also emitted under
        # that key, [window_size,3,H,W] in the wan_frames_range; wan_frames stays the centered `wan_num_frames` exposure slice.
        self.ctx_key = opt.get("ctx_key", None)

        # Optional Wan-video-clip target: a second, centered sub-window of
        # the same real window, independent of vjepa_window_size (a Wan VAE
        # needs an odd `4k+1` frame count, while V-JEPA needs an even one --
        # the two constraints are incompatible on a single slice). Additive
        # and off by default so existing BlurJepaDynamics.yml runs are
        # unaffected.
        self.include_wan_frames = bool(opt.get("include_wan_frames", False))
        self.wan_num_frames = int(opt.get("wan_num_frames", 9))
        self.wan_vae_scale_factor_temporal = int(opt.get("wan_vae_scale_factor_temporal", 4))
        self.wan_prefix_frames = int(opt.get("wan_prefix_frames", 0))
        self.wan_key = opt.get("wan_key", "wan_frames")
        self.wan_frames_range = opt.get("wan_frames_range", "pm1")

        # burst_size > 1: BURST mode -- each returned sample groups `burst_size`
        # CONSECUTIVE listed blur centers (from the same scene) into one
        # temporally-ordered sequence, instead of the single blur image the
        # rest of this class returns by default. Purely additive: burst_size=1
        # (the default) leaves every existing code path in this file
        # byte-identical -- see __getitem__'s dispatch and the burst-only
        # sample discovery below. Motivation (see findings/ for the full
        # writeup): a SINGLE blurred image is time-symmetric (its blur trail
        # looks identical played forward or backward), which is the root
        # cause of the persistent Stage-C reversal-rate problem in the
        # single-frame line of work -- a burst of `burst_size` blur images,
        # each averaging its own distinct, temporally-ordered sub-window,
        # is NOT time-symmetric as a sequence (segment i's content precedes
        # segment i+1's by construction), which should make the correct
        # generation direction recoverable from the input itself rather than
        # depending on a learned prior.
        #
        # Reuses the EXISTING precomputed per-center blur files directly (no
        # new blur synthesis) -- verified spacing between consecutive listed
        # centers in this dataset (8 frames) is smaller than window_size
        # (11), so consecutive segments' 11-frame averaging windows overlap
        # by a few frames; this is harmless (each segment's blur image still
        # predominantly reflects its own local motion) and requires no new
        # image-processing code.
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

        extensions = tuple(opt.get("extensions", [".png", ".jpg", ".jpeg"]))

        full_sharp_dir = os.path.join(self.data_dir, "full_sharp")
        blur_dir = os.path.join(self.data_dir, f"{self.split}_blur")
        list_dir = os.path.join(self.data_dir, f"{self.split}_list")

        scenes = sorted(
            f[: -len("_im_list.txt")]
            for f in os.listdir(list_dir)
            if f.endswith("_im_list.txt")
        )
        # Each entry additionally carries center_idx/ext/scene_full_sharp
        # (beyond blur_path/frame_paths) so burst mode below can reconstruct
        # a CONTINUOUS multi-segment real-frame span without re-deriving
        # scene layout -- unused by the burst_size=1 path, which only reads
        # the first two fields.
        per_scene_samples: List[List[Tuple[str, List[str], int, str, str]]] = []
        for scene in scenes:
            scene_full_sharp = os.path.join(full_sharp_dir, scene)
            scene_blur = os.path.join(blur_dir, scene)
            list_path = os.path.join(list_dir, f"{scene}_im_list.txt")
            with open(list_path) as f:
                center_names = [line.strip() for line in f if line.strip()]

            scene_samples: List[Tuple[str, List[str], int, str, str]] = []
            for name in center_names:
                if not name.lower().endswith(extensions):
                    continue
                stem, ext = os.path.splitext(name)
                try:
                    center_idx = int(stem)
                except ValueError:
                    continue

                frame_paths = [
                    os.path.join(scene_full_sharp, f"{center_idx - self.half_window + i:05d}{ext}")
                    for i in range(self.window_size)
                ]
                blur_path = os.path.join(scene_blur, name)
                if not os.path.isfile(blur_path):
                    continue
                if not all(os.path.isfile(p) for p in frame_paths):
                    # Too close to the scene's start/end for a full window.
                    continue
                scene_samples.append((blur_path, frame_paths, center_idx, ext, scene_full_sharp))
            if scene_samples:
                per_scene_samples.append(scene_samples)

        # Interleave round-robin across scenes (rather than concatenating
        # scene-by-scene) so the first N samples in *unshuffled* iteration
        # order span N different scenes -- this repo's convention is
        # use_shuffle: false for validation, so with a small max_val_steps
        # every validation cycle would otherwise draw exclusively from
        # whichever scene happens to sort first.
        if self.burst_size == 1:
            self.samples: List[Tuple[str, List[str]]] = [
                (s[0], s[1])
                for group in zip_longest(*per_scene_samples)
                for s in group
                if s is not None
            ]
        else:
            # BURST mode: group each scene's own listed centers into
            # NON-OVERLAPPING chunks of burst_size consecutive centers
            # (stride=burst_size) *before* interleaving across scenes, so no
            # two bursts share a segment and every scene still contributes
            # independently-ordered chunks (mirrors the burst_size=1
            # interleave above, just one level up: group-of-groups instead
            # of group-of-samples). A trailing partial chunk (fewer than
            # burst_size centers left) is dropped.
            per_scene_bursts: List[List[List[Tuple[str, List[str], int, str, str]]]] = []
            for scene_samples in per_scene_samples:
                bursts = [
                    scene_samples[start:start + self.burst_size]
                    for start in range(0, len(scene_samples) - self.burst_size + 1, self.burst_size)
                ]
                if bursts:
                    per_scene_bursts.append(bursts)
            self.samples: List[List[Tuple[str, List[str], int, str, str]]] = [
                b for group in zip_longest(*per_scene_bursts) for b in group if b is not None
            ]

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid Adobe240 samples found under {self.data_dir} for split={self.split!r} "
                f"(burst_size={self.burst_size})."
            )
        print(
            f"Initialized Adobe240Dataset ({self.split}) with {len(self.samples)} samples "
            f"across {len(scenes)} scenes (window_size={self.window_size}, "
            f"vjepa_window_size={self.vjepa_window_size}, burst_size={self.burst_size})"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load01(path: str) -> np.ndarray:
        """[H,W,3] float32 in [0,1]."""
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    @staticmethod
    def _load01_resized(path: str, w: int, h: int) -> np.ndarray:
        """[h,w,3] float32 in [0,1], whole-frame resized (non-aspect-preserving,
        matching blur2vid's inference.py::convert_to_batch convention:
        `img.resize((width, height))` with PIL's default resampling) rather
        than cropped."""
        img = Image.open(path).convert("RGB").resize((w, h))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _sample_crop_hw(self, native_h: int, native_w: int) -> Tuple[int, int]:
        """(crop_h, crop_w) with area ~= crop_size^2. Degenerates to exactly
        (crop_size, crop_size) -- the original fixed-square-crop behavior --
        whenever aspect_ratio_range is unset or random_crop is False (e.g.
        validation). Otherwise draws crop_w/crop_h uniformly from
        aspect_ratio_range, clamps to fit within (native_h, native_w)
        (rebalancing the other dimension to preserve area as closely as
        possible when one side has to shrink to fit), then rounds each dim
        down to the nearest multiple of crop_align."""
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
        """True when the crop shape must be picked once per *batch* (so all
        samples in the batch share the same tensor shape to stack) rather
        than independently per sample."""
        return self.resize_mode == "crop" and self.random_crop and self.aspect_ratio_range is not None

    def _to_out_size(self, arr01: np.ndarray) -> np.ndarray:
        """Resize [H,W,3] or [T,H,W,3] in [0,1] to (out_size, out_size).
        No-op when out_size is unset or already matches. Uses PIL with the
        same default resampling as _load01_resized, so every resize path in
        this file stays consistent."""
        if self.out_size is None:
            return arr01
        if arr01.shape[-3] == self.out_size and arr01.shape[-2] == self.out_size:
            return arr01
        if arr01.ndim == 3:
            img = Image.fromarray((arr01 * 255.0).astype(np.uint8))
            img = img.resize((self.out_size, self.out_size))
            return np.asarray(img, dtype=np.float32) / 255.0
        return np.stack([self._to_out_size(f) for f in arr01])

    def _finalize_sample(self, blur01: np.ndarray, frames01_full: np.ndarray, blur_path: str) -> dict:
        """blur01: [H,W,3] in [0,1] (already cropped/resized). frames01_full:
        [window_size,H,W,3] in [0,1], same crop region as blur01. Builds the
        final returned sample dict -- shared by both the normal per-sample
        path and the deferred (aspect-ratio-batched) collate_fn path."""
        blur01 = self._to_out_size(blur01)
        frames01_full = self._to_out_size(frames01_full)
        blur_pm1 = (blur01.transpose(2, 0, 1) * 2.0 - 1.0).astype(np.float32)

        sample = {
            self.lq_key: blur_pm1,
        }

        if self.include_wan_frames:
            # Centered sub-window of the same real window, independent of
            # the vjepa_window_size prefix-slice below.
            wan_start = (self.window_size - self.wan_num_frames) // 2
            wan_frames01 = frames01_full[wan_start:wan_start + self.wan_num_frames]
            wan_frames01 = wan_frames01.transpose(0, 3, 1, 2).astype(np.float32)  # [T,3,H,W]
            if self.wan_frames_range == "pm1":
                wan_frames01 = wan_frames01 * 2.0 - 1.0
            sample[self.wan_key] = wan_frames01
            if self.ctx_key:
                ctx01 = frames01_full.transpose(0, 3, 1, 2).astype(np.float32)
                sample[self.ctx_key] = ctx01 * 2.0 - 1.0 if self.wan_frames_range == "pm1" else ctx01

        frames01 = frames01_full[: self.vjepa_window_size]  # even prefix for V-JEPA
        frames01 = frames01.transpose(0, 3, 1, 2).astype(np.float32)  # [T,3,H,W]
        sample[self.gt_key] = frames01

        if self.return_name:
            # Every scene's blur-frame filenames restart from a similar index
            # (e.g. every scene's first valid center frame is "00017.png"),
            # so the bare basename collides across scenes -- prefix with the
            # scene name (no "/" separator, so downstream os.path.basename()
            # calls on this name leave it intact) to keep validation
            # image/video names -- and hence disk files and Comet assets --
            # unique once samples are interleaved across scenes.
            scene = os.path.basename(os.path.dirname(blur_path))
            sample["name"] = f"{scene}_{os.path.basename(blur_path)}"
        return sample

    def collate_fn(self, batch: List[dict]):
        """Default-collates a batch, except when aspect_ratio_range deferred
        cropping to here (see _defer_crop_to_collate): picks ONE (crop_h,
        crop_w) for the whole batch, so every sample in it ends up the same
        shape, then finalizes each sample with that shared crop. Falls back
        to plain default_collate whenever nothing was deferred (the common
        case -- validation, resize_mode="resize", or aspect_ratio_range
        unset), so this is always safe to wire in as the DataLoader's
        collate_fn regardless of config. Handles both the single-frame
        (_native_blur01, [H,W,3]) and burst-mode (_native_burst_blur01,
        [S,H,W,3]) deferred shapes."""
        if not batch:
            return default_collate(batch)

        if "_native_burst_blur01" in batch[0]:
            native_h = min(item["_native_burst_blur01"].shape[1] for item in batch)
            native_w = min(item["_native_burst_blur01"].shape[2] for item in batch)
            crop_h, crop_w = self._sample_crop_hw(native_h, native_w)
            finalized = []
            for item in batch:
                burst_blur01_native = item["_native_burst_blur01"]  # [S,H,W,3]
                full_frames01_native = item["_native_burst_full_frames01"]  # [span,H,W,3]
                h, w = burst_blur01_native.shape[1:3]
                ch, cw = min(crop_h, h), min(crop_w, w)
                top, left = self._crop_region(h, w, ch, cw)
                burst_blur01 = burst_blur01_native[:, top:top + ch, left:left + cw]
                full_frames01 = full_frames01_native[:, top:top + ch, left:left + cw]
                finalized.append(self._finalize_burst_sample(
                    burst_blur01, full_frames01, item["_burst_centers"],
                    item["_burst_full_span_start"], item["_burst_first_blur_path"],
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
            blur01 = blur01_native[top:top + ch, left:left + cw]
            frames01_full = frames01_full_native[:, top:top + ch, left:left + cw]
            finalized.append(self._finalize_sample(blur01, frames01_full, item["_blur_path"]))
        return default_collate(finalized)

    def _finalize_burst_sample(
        self,
        burst_blur01: np.ndarray,
        full_frames01: np.ndarray,
        centers: List[int],
        full_span_start: int,
        first_blur_path: str,
    ) -> dict:
        """burst_blur01: [burst_size,H,W,3] in [0,1] -- one pre-synthesized
        blur image per segment, temporally ordered (segment i precedes
        segment i+1). full_frames01: [full_span_len,H,W,3] in [0,1] -- the
        CONTINUOUS real-frame span covering every segment's own
        window_size-length averaging window end-to-end, i.e. from
        centers[0]-half_window to centers[-1]+half_window inclusive. Loaded
        once (by the caller) rather than per-segment since consecutive
        segments' windows overlap. centers: the burst_size listed center
        frame indices, used here only to locate each segment's own window
        inside full_frames01. Builds the final burst-mode sample dict."""
        burst_blur01 = self._to_out_size(burst_blur01)
        full_frames01 = self._to_out_size(full_frames01)
        burst_blur_pm1 = (burst_blur01.transpose(0, 3, 1, 2) * 2.0 - 1.0).astype(np.float32)  # [B,3,H,W]
        sample = {self.burst_key: burst_blur_pm1}

        if self.include_wan_frames:
            # The real video Wan is asked to generate: a centered
            # wan_num_frames-length sub-window of the FULL burst span (the
            # whole temporally-ordered sequence), NOT any single segment's
            # local window.
            full_span_len = full_frames01.shape[0]
            wan_start = (full_span_len - self.wan_num_frames) // 2
            if wan_start < 0:
                raise ValueError(
                    f"wan_num_frames={self.wan_num_frames} exceeds the full burst span "
                    f"({full_span_len} frames, burst_size={self.burst_size}); reduce "
                    f"wan_num_frames or increase burst_size/window_size."
                )
            wan_frames01 = full_frames01[wan_start:wan_start + self.wan_num_frames]
            wan_frames01 = wan_frames01.transpose(0, 3, 1, 2).astype(np.float32)  # [T,3,H,W]
            if self.wan_frames_range == "pm1":
                wan_frames01 = wan_frames01 * 2.0 - 1.0
            sample[self.wan_key] = wan_frames01

        # Per-segment V-JEPA-ready frames: for each segment, the even
        # vjepa_window_size-length prefix of its own window_size window,
        # sliced out of the shared full_frames01 span (each segment's window
        # lies entirely inside that span by construction). Stage-1 oracle
        # mode runs frozen V-JEPA on each segment independently, then
        # concatenates the resulting per-segment token grids along the
        # tubelet axis -- see Wan22VJEPADynamics_model.py's burst-aware
        # _dynamics_full_grid branch.
        burst_frames = []
        for c in centers:
            seg_start = (c - self.half_window) - full_span_start
            seg_window = full_frames01[seg_start:seg_start + self.window_size]
            burst_frames.append(seg_window[: self.vjepa_window_size])  # even prefix for V-JEPA
        burst_frames = np.stack(burst_frames)  # [burst_size, vjepa_window_size, H, W, 3]
        burst_frames = burst_frames.transpose(0, 1, 4, 2, 3).astype(np.float32)  # [B,T,3,H,W]
        sample[self.burst_frames_key] = burst_frames

        if self.return_name:
            scene = os.path.basename(os.path.dirname(first_blur_path))
            sample["name"] = f"{scene}_{os.path.basename(first_blur_path)}_burst{len(centers)}"
        return sample

    def _getitem_burst(self, idx):
        burst = self.samples[idx]  # list of burst_size (blur_path, frame_paths, center_idx, ext, scene_full_sharp)
        centers = [b[2] for b in burst]
        ext = burst[0][3]
        scene_full_sharp = burst[0][4]
        full_span_start = centers[0] - self.half_window
        full_span_end = centers[-1] + self.half_window
        full_frame_paths = [
            os.path.join(scene_full_sharp, f"{i:05d}{ext}")
            for i in range(full_span_start, full_span_end + 1)
        ]
        blur_paths = [b[0] for b in burst]

        if self.resize_mode == "resize":
            burst_blur01 = np.stack([
                self._load01_resized(p, self.resize_w, self.resize_h) for p in blur_paths
            ])
            full_frames01 = np.stack([
                self._load01_resized(p, self.resize_w, self.resize_h) for p in full_frame_paths
            ])
            return self._finalize_burst_sample(
                burst_blur01, full_frames01, centers, full_span_start, blur_paths[0]
            )

        blur01_native_list = [self._load01(p) for p in blur_paths]
        if self._defer_crop_to_collate():
            # Mirrors the single-frame deferred path (see __getitem__): the
            # crop shape depends on what the rest of this batch's native
            # sizes allow, which isn't known here -- collate_fn finishes
            # this sample once the whole batch is assembled.
            full_frames01_native = np.stack([self._load01(p) for p in full_frame_paths])
            return {
                "_native_burst_blur01": np.stack(blur01_native_list),  # [S,H,W,3]
                "_native_burst_full_frames01": full_frames01_native,  # [span,H,W,3]
                "_burst_centers": centers,
                "_burst_full_span_start": full_span_start,
                "_burst_first_blur_path": blur_paths[0],
            }

        # Direct per-sample crop -- same crop region applied to every burst
        # blur image and every frame in the full continuous span.
        h, w = blur01_native_list[0].shape[:2]
        crop_h, crop_w = self._sample_crop_hw(h, w)
        ch, cw = min(crop_h, h), min(crop_w, w)
        top, left = self._crop_region(h, w, ch, cw)
        burst_blur01 = np.stack([b[top:top + ch, left:left + cw] for b in blur01_native_list])
        full_frames01 = np.stack([
            self._load01(p)[top:top + ch, left:left + cw] for p in full_frame_paths
        ])

        return self._finalize_burst_sample(
            burst_blur01, full_frames01, centers, full_span_start, blur_paths[0]
        )

    def __getitem__(self, idx):
        if self.burst_size > 1:
            return self._getitem_burst(idx)

        blur_path, frame_paths = self.samples[idx]

        if self.resize_mode == "resize":
            # Whole-frame resize (no crop) -- every frame in the window
            # already describes the same physical scene region.
            blur01 = self._load01_resized(blur_path, self.resize_w, self.resize_h)
            frames01_full = np.stack([
                self._load01_resized(p, self.resize_w, self.resize_h) for p in frame_paths
            ])  # [window_size, H, W, 3] in [0,1]
            return self._finalize_sample(blur01, frames01_full, blur_path)

        blur01_native = self._load01(blur_path)
        if self._defer_crop_to_collate():
            # The crop shape depends on what the rest of this batch's native
            # sizes allow, which isn't known here -- collate_fn finishes
            # this sample once the whole batch is assembled.
            frames01_full_native = np.stack([self._load01(p) for p in frame_paths])
            return {
                "_native_blur01": blur01_native,
                "_native_frames01": frames01_full_native,
                "_blur_path": blur_path,
            }

        h, w = blur01_native.shape[:2]
        crop_h, crop_w = self._sample_crop_hw(h, w)
        ch, cw = min(crop_h, h), min(crop_w, w)
        top, left = self._crop_region(h, w, ch, cw)

        # Same crop region applied to the blur image and every frame in
        # the window, so the V-JEPA target and the blurred input
        # describe the same physical scene region.
        blur01 = blur01_native[top:top + ch, left:left + cw]
        frames01_full = np.stack([
            self._load01(p)[top:top + ch, left:left + cw] for p in frame_paths
        ])  # [window_size, H, W, 3] in [0,1]
        return self._finalize_sample(blur01, frames01_full, blur_path)

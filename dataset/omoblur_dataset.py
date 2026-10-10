"""OMoBlurDataset: pairs a pre-synthesized blurred frame from OMoBlur with the
sharp frames of its own exposure window, mirroring Adobe240Dataset's
interface (single-sample and burst modes). Masks (`*_m.png`) are ignored unless
`mask_crop: true` (see below).

mask_crop (train, random_crop): `<c>_<N>_m.png` is a binary (0/255) map of where the blur is
(median ~23% of the frame, ~5% of masks nearly empty, some 100%). With mask_crop the crop's
horizontal position is chosen so at least `mask_min_coverage` of the crop is blurred (uniform over
all such positions; best-available if none qualifies), EXCEPT with probability
`mask_blur_free_prob` (default 0.1), when a (near-)blur-free crop (<= `mask_blur_free_max_coverage`
mask coverage) is drawn instead, so the model still sees sharp-ish content. If no blur-free
position exists the sample falls back to an ordinary random crop. Burst mode uses the union of the
segments' masks. With random_crop false (val) the crop is the deterministic max-coverage window.
Not combinable with crop_offsets or resize_mode='resize'.

Layout at data_dir (e.g. /scratch/gilbreth/wweligam/OMoBlur), already extracted:
    <split>/<group>/<seq>/image_sequence/001.png ... NNN.png   sharp frames
    <split>/<group>/<seq>/image_pairs/<c>_1.png                 sharp frame c (unused)
    <split>/<group>/<seq>/image_pairs/<c>_<N>_b.png             blur of N frames
    <split>/<group>/<seq>/image_pairs/<c>_<N>_m.png             mask (unused)

Verified directly: `<c>_<N>_b.png` is the average of the N-frame window
CENTERED on sequence frame c (frames c-N//2 .. c+N//2, 1-indexed; mean abs
error vs. the mean of those frames was lowest for that window for
N=7/11/15). Blur centers within a sequence are spaced 16 frames apart
(8, 24, 40), so for N <= 15 consecutive windows never overlap.

Native resolution: 1920x360, a 5.33:1 aspect ratio -- much wider than this
project's other sources (Adobe240/GoPro/FMO, all square or ~16:9 crops).
Splits actually present on disk: "train" (16 groups, ~2100 sequences) and
"val" (16 groups, ~145 sequences, very uneven per group: 1 to 29). There is
NO on-disk "test" split -- analysis/make_omoblur_test_subset.py carves a
fixed, reproducible, diversity-first subset out of "val" for that purpose
(see that script's docstring for the selection rule).

wan_frames (include_wan_frames=True): a Wan-video generation target, sized
independently of the blur length N, read from the wider `image_sequence`
pool (up to the whole 48-frame sequence) rather than the blur's own N-frame
averaging window -- mirrors Adobe240Dataset's single-frame-vs-burst split:
  * single-frame (burst_size=1): centered on this sample's own blur center,
    clamped to the sequence's frame range. Sequences are only 48 frames with
    edge centers (8, 40) just 7 frames from the boundary, so a symmetric
    window fitting EVERY center tops out at wan_num_frames=15 -- larger
    values silently keep only center=24 samples (filtered at init, not a
    crash; see the "wan_num_frames dropped N samples" log line).
  * burst (burst_size>1): spans the full burst (first center - half_window
    to last center + half_window), same pattern as Adobe240Dataset's burst
    path, so a 3-segment burst (centers 8/24/40) can support a much longer
    window (up to ~47 frames) since it already spans nearly the sequence.

crop_offsets (width diversity): OMoBlur's frames are wide enough that a
single crop_size-square crop only ever samples ONE horizontal slice. Set
crop_offsets to a list of floats in [0,1] (0=left edge, 1=right edge) to
expand EVERY sample into len(crop_offsets) separate, fixed, reproducible
crops at those horizontal positions -- "crop from different widths" for a
diverse, deterministic val/test set. None (default): unchanged behavior,
one crop per sample (random position when random_crop=True, else centered).

Returned sample (burst_size=1):
    lq_key  ("blur")   [3,H,W]    float32 in [-1,1]
    gt_key  ("frames") [V,3,H,W]  float32 in [0,1]  (V = vjepa_window_size, an EVEN PREFIX of the
                                  blur_lengths window -- NOT the full N frames; see
                                  self.vjepa_window_size's fix note, this is what V-JEPA sees)
    wan_key ("wan_frames") [F,3,H,W] float32, range per wan_frames_range   (if include_wan_frames)
Returned sample (burst_size>1), temporally ordered consecutive centers of one sequence:
    burst_key        ("burst_blur")   [S,3,H,W]    in [-1,1]
    burst_frames_key ("burst_frames") [S,V,3,H,W]  in [0,1]  (V = vjepa_window_size, see above)
    wan_key ("wan_frames") [F,3,H,W] float32, range per wan_frames_range   (if include_wan_frames)
"""
import os
import random
import re
from itertools import zip_longest
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils import data as data

_BLUR_RE = re.compile(r"^(\d+)_(\d+)_b\.png$")


class OMoBlurDataset(data.Dataset):
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
        out_size = opt.get("out_size", None)
        self.out_size = int(out_size) if out_size is not None else None

        # "crop": square crop_size crop. "resize": whole-frame resize to (resize_h, resize_w).
        self.resize_mode = opt.get("resize_mode", "crop")
        if self.resize_mode not in ("crop", "resize"):
            raise ValueError(f"Unknown resize_mode: {self.resize_mode!r}")
        self.resize_h = int(opt.get("resize_h", 360))
        self.resize_w = int(opt.get("resize_w", 1920))

        # crop_offsets: see module docstring. None -> old single-crop-per-sample behavior.
        crop_offsets = opt.get("crop_offsets", None)
        self.crop_offsets = [float(x) for x in crop_offsets] if crop_offsets is not None else None
        if self.crop_offsets is not None:
            if not self.crop_offsets:
                raise ValueError("crop_offsets, if given, must be non-empty")
            if any(not (0.0 <= x <= 1.0) for x in self.crop_offsets):
                raise ValueError(f"crop_offsets must all be in [0,1], got {self.crop_offsets!r}")
            if self.random_crop:
                raise ValueError(
                    "crop_offsets is for deterministic, reproducible multi-width sampling (val/test) "
                    "and conflicts with random_crop -- set random_crop: false to use it."
                )

        self.mask_crop = bool(opt.get("mask_crop", False))
        self.mask_min_coverage = float(opt.get("mask_min_coverage", 0.25))
        self.mask_blur_free_prob = float(opt.get("mask_blur_free_prob", 0.1))
        self.mask_blur_free_max_coverage = float(opt.get("mask_blur_free_max_coverage", 0.02))
        if self.mask_crop and (self.crop_offsets is not None or self.resize_mode != "crop"):
            raise ValueError("mask_crop needs resize_mode='crop' and no crop_offsets.")
        if not 0.0 <= self.mask_blur_free_prob <= 1.0:
            raise ValueError(f"mask_blur_free_prob must be in [0,1], got {self.mask_blur_free_prob}")

        # Which blur lengths (exposure windows) to expose; each (center, N)
        # pair is its own sample. Bursts never mix lengths.
        self.blur_lengths = [int(n) for n in list(opt.get("blur_lengths", [7, 11, 15]))]

        self.burst_size = int(opt.get("burst_size", 1))
        if self.burst_size < 1:
            raise ValueError(f"burst_size must be >= 1, got {self.burst_size}")
        self.burst_key = opt.get("burst_key", "burst_blur")
        self.burst_frames_key = opt.get("burst_frames_key", "burst_frames")

        # BUG FIXED 2026-09-28: `frames`/`burst_frames` (the gt_key/burst_frames_key sample) is
        # what a dynamics_source="vjepa_gt" run feeds to V-JEPA directly (see
        # Wan22VJEPADynamics_model._oracle_grid_burst -- it uses whatever frame count the dataset
        # hands it, with no truncation of its own). Every OTHER dataset here separates the full
        # blur-reconstruction window from an even, SHORTER vjepa_window_size prefix actually fed to
        # V-JEPA (see Adobe240Dataset's module docstring; FallingObjectsDataset literally does
        # `seg_window[: self.vjepa_window_size]`) -- this class never did, so e.g. blur_lengths=[15]
        # fed all 15 (odd -> V-JEPA's tubelet_size=2 truncates to 14) frames per segment, giving
        # 15//2=7 tubelets where every other source in the same mix gives 4 (vjepa_window_size=8//
        # 2), and crashed cross-attention's fixed-size positional table the first time this class
        # was actually plugged into that code path (never exercised before). vjepa_window_size
        # must be <= the smallest configured blur_lengths value.
        self.vjepa_window_size = int(opt.get("vjepa_window_size", 8))
        # v3 context frames: `ctx_frames` (e.g. 19) real consecutive frames centered on the blur center are loaded and emitted under
        # `ctx_key`; wan_frames becomes the centered `wan_num_frames` slice of them. Sequences too short for the window are dropped.
        self.ctx_key = opt.get("ctx_key", None)
        self.ctx_frames = int(opt.get("ctx_frames", 0)) if self.ctx_key else 0
        if self.vjepa_window_size > min(self.blur_lengths):
            raise ValueError(
                f"vjepa_window_size={self.vjepa_window_size} exceeds the smallest blur_lengths "
                f"value ({min(self.blur_lengths)}) -- frames/burst_frames only has that many."
            )

        # Optional Wan-video-clip target -- see module docstring. Independent of blur_lengths;
        # read from the wider image_sequence pool, not the blur's own N-frame averaging window.
        self.include_wan_frames = bool(opt.get("include_wan_frames", False))
        self.wan_num_frames = int(opt.get("wan_num_frames", 15))
        self.wan_key = opt.get("wan_key", "wan_frames")
        self.wan_frames_range = opt.get("wan_frames_range", "pm1")
        if self.wan_frames_range not in ("pm1", "01"):
            raise ValueError(f"Unknown wan_frames_range: {self.wan_frames_range!r}")

        split_dir = os.path.join(self.data_dir, self.split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"OMoBlur split dir not found: {split_dir}")

        # Per (sequence, blur length): list of (blur_path, center, n, seq_dir, n_seq_frames).
        per_seq: List[List[Tuple[str, int, int, str, int]]] = []
        dropped_for_wan = 0
        for group in sorted(os.listdir(split_dir)):
            group_dir = os.path.join(split_dir, group)
            if not os.path.isdir(group_dir):
                continue
            for seq in sorted(os.listdir(group_dir)):
                pairs_dir = os.path.join(group_dir, seq, "image_pairs")
                seq_frames_dir = os.path.join(group_dir, seq, "image_sequence")
                if not (os.path.isdir(pairs_dir) and os.path.isdir(seq_frames_dir)):
                    continue
                by_len = {n: [] for n in self.blur_lengths}
                # One listdir per sequence instead of a stat per frame (slow on scratch).
                available = set(os.listdir(seq_frames_dir))
                n_seq_frames = len(available)
                for fname in os.listdir(pairs_dir):
                    m = _BLUR_RE.match(fname)
                    if m is None:
                        continue
                    center, n = int(m.group(1)), int(m.group(2))
                    if n not in by_len:
                        continue
                    half = n // 2
                    if not all(f"{center - half + i:03d}.png" in available for i in range(n)):
                        continue
                    by_len[n].append((os.path.join(pairs_dir, fname), center, n, os.path.join(group_dir, seq), n_seq_frames))
                for n in self.blur_lengths:
                    if by_len[n]:
                        per_seq.append(sorted(by_len[n], key=lambda s: s[1]))

        # Round-robin across sequences so the first N unshuffled samples span
        # N different sequences (validation convention in this repo).
        if self.burst_size == 1:
            base_samples = [s for grp in zip_longest(*per_seq) for s in grp if s is not None]
            if self.include_wan_frames:
                kept = []
                for s in base_samples:
                    center, n_seq_frames = s[1], s[4]
                    half_f = max(self.wan_num_frames, self.ctx_frames) // 2
                    if center - half_f >= 1 and center + half_f <= n_seq_frames:
                        kept.append(s)
                    else:
                        dropped_for_wan += 1
                base_samples = kept
            self.samples = base_samples
        else:
            per_seq_bursts = []
            for seq_samples in per_seq:
                bursts = [
                    seq_samples[i:i + self.burst_size]
                    for i in range(0, len(seq_samples) - self.burst_size + 1, self.burst_size)
                ]
                if bursts:
                    per_seq_bursts.append(bursts)
            base_samples = [b for grp in zip_longest(*per_seq_bursts) for b in grp if b is not None]
            if self.include_wan_frames:
                kept = []
                for b in base_samples:
                    n_seq_frames = b[0][4]
                    full_span_start = b[0][1] - (b[0][2] // 2)
                    full_span_end = b[-1][1] + (b[-1][2] // 2)
                    full_span_len = full_span_end - full_span_start + 1
                    if full_span_start >= 1 and full_span_end <= n_seq_frames and full_span_len >= self.wan_num_frames:
                        kept.append(b)
                    else:
                        dropped_for_wan += 1
                base_samples = kept
            self.samples = base_samples

        # crop_offsets multiplies every remaining sample into one entry per offset.
        if self.crop_offsets is not None:
            self.samples = [(s, off) for s in self.samples for off in self.crop_offsets]
        else:
            self.samples = [(s, None) for s in self.samples]

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid OMoBlur samples under {self.data_dir} for split={self.split!r} "
                f"(blur_lengths={self.blur_lengths}, burst_size={self.burst_size}, "
                f"include_wan_frames={self.include_wan_frames}, wan_num_frames={self.wan_num_frames})."
            )
        wan_msg = ""
        if self.include_wan_frames and dropped_for_wan:
            wan_msg = f"; wan_num_frames={self.wan_num_frames} dropped {dropped_for_wan} sequence groups that didn't fit"
        print(
            f"Initialized OMoBlurDataset ({self.split}) with {len(self.samples)} samples "
            f"across {len(per_seq)} sequence/length groups "
            f"(blur_lengths={self.blur_lengths}, burst_size={self.burst_size}, "
            f"crop_offsets={self.crop_offsets}){wan_msg}"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _window_paths(seq_dir: str, center: int, n: int) -> List[str]:
        half = n // 2
        return [
            os.path.join(seq_dir, "image_sequence", f"{center - half + i:03d}.png")
            for i in range(n)
        ]

    @staticmethod
    def _wan_window_paths(seq_dir: str, center: int, n_frames: int) -> List[str]:
        half = n_frames // 2
        return [
            os.path.join(seq_dir, "image_sequence", f"{center - half + i:03d}.png")
            for i in range(n_frames)
        ]

    def _load01(self, path: str) -> np.ndarray:
        """[H,W,3] float32 in [0,1]; whole-frame resized when resize_mode='resize'."""
        img = Image.open(path).convert("RGB")
        if self.resize_mode == "resize":
            img = img.resize((self.resize_w, self.resize_h))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _crop_region(self, h: int, w: int, ch: int, cw: int, offset: Optional[float]) -> Tuple[int, int]:
        if h < ch or w < cw:
            return 0, 0
        top = (h - ch) // 2  # OMoBlur frames are wide, not tall -- height is never varied.
        if offset is not None:
            # Deterministic width position: offset=0 -> left edge, 1 -> right edge.
            left = int(round(offset * (w - cw)))
            return top, left
        if self.random_crop:
            return random.randint(0, h - ch), random.randint(0, w - cw)
        return top, (w - cw) // 2

    @staticmethod
    def _mask_path(blur_path: str) -> str:
        return blur_path[: -len("_b.png")] + "_m.png"

    def _load_mask(self, blur_paths: List[str]) -> Optional[np.ndarray]:
        """Union of the blur masks, [H,W] bool; None if any is missing."""
        union = None
        for bp in blur_paths:
            mp = self._mask_path(bp)
            if not os.path.exists(mp):
                return None
            m = np.asarray(Image.open(mp).convert("L")) > 127
            union = m if union is None else (union | m)
        return union

    def _mask_left(self, mask: np.ndarray, top: int, ch: int, cw: int) -> Optional[int]:
        """Pick the crop's left edge from mask coverage; None -> caller falls back."""
        h, w = mask.shape
        if h < top + ch or w < cw:
            return None
        cols = mask[top:top + ch].sum(0).astype(np.float64)
        csum = np.concatenate([[0.0], np.cumsum(cols)])
        cov = (csum[cw:] - csum[:-cw]) / float(ch * cw)  # coverage for every left in [0, w-cw]
        if not self.random_crop:
            return int(np.argmax(cov))
        if random.random() < self.mask_blur_free_prob:
            free = np.flatnonzero(cov <= self.mask_blur_free_max_coverage)
            return int(random.choice(free)) if len(free) else None
        ok = np.flatnonzero(cov >= self.mask_min_coverage)
        return int(random.choice(ok)) if len(ok) else int(np.argmax(cov))

    def _to_out_size(self, arr01: np.ndarray) -> np.ndarray:
        """Resize [...,H,W,3] in [0,1] to (out_size, out_size); no-op if unset."""
        if self.out_size is None:
            return arr01
        if arr01.shape[-3] == self.out_size and arr01.shape[-2] == self.out_size:
            return arr01
        if arr01.ndim == 3:
            img = Image.fromarray((arr01 * 255.0).astype(np.uint8)).resize((self.out_size, self.out_size))
            return np.asarray(img, dtype=np.float32) / 255.0
        return np.stack([self._to_out_size(f) for f in arr01])

    def _load_group(self, blur_paths: List[str], frame_path_lists: List[List[str]], offset: Optional[float],
                     wan_paths: Optional[List[str]] = None):
        """Load blur images [S,H,W,3], per-segment windows [S,N,H,W,3] and optionally the
        wan-target window [F,H,W,3], with ONE shared crop region, then apply out_size."""
        blurs = [self._load01(p) for p in blur_paths]
        frames = [[self._load01(p) for p in fps] for fps in frame_path_lists]
        wan = [self._load01(p) for p in wan_paths] if wan_paths is not None else None
        h, w = blurs[0].shape[:2]
        if self.resize_mode == "crop":
            ch, cw = min(self.crop_size, h), min(self.crop_size, w)
            top, left = self._crop_region(h, w, ch, cw, offset)
            if self.mask_crop:
                mask = self._load_mask(blur_paths)
                mleft = self._mask_left(mask, top, ch, cw) if mask is not None else None
                if mleft is not None:
                    left = mleft
            blurs = [b[top:top + ch, left:left + cw] for b in blurs]
            frames = [[f[top:top + ch, left:left + cw] for f in seg] for seg in frames]
            if wan is not None:
                wan = [f[top:top + ch, left:left + cw] for f in wan]
        blurs = self._to_out_size(np.stack(blurs))
        frames = np.stack([self._to_out_size(np.stack(seg)) for seg in frames])
        wan = self._to_out_size(np.stack(wan)) if wan is not None else None
        return blurs, frames, wan

    def _name(self, blur_path: str) -> str:
        # .../<split>/<group>/<seq>/image_pairs/<c>_<N>_b.png -> "<group>_<seq>_<c>_<N>"
        pairs_dir = os.path.dirname(blur_path)
        seq_dir = os.path.dirname(pairs_dir)
        group = os.path.basename(os.path.dirname(seq_dir))
        stem = os.path.basename(blur_path)[: -len("_b.png")]
        return f"{group}_{os.path.basename(seq_dir)}_{stem}"

    def __getitem__(self, idx):
        entry, offset = self.samples[idx]
        suffix = f"_w{offset:.2f}" if offset is not None else ""

        if self.burst_size > 1:
            burst = entry
            wan_paths = None
            if self.include_wan_frames:
                seq_dir0 = burst[0][3]
                full_span_start = burst[0][1] - (burst[0][2] // 2)
                full_span_end = burst[-1][1] + (burst[-1][2] // 2)
                full_span_len = full_span_end - full_span_start + 1
                wan_start = full_span_start + (full_span_len - self.wan_num_frames) // 2
                wan_paths = [
                    os.path.join(seq_dir0, "image_sequence", f"{wan_start + i:03d}.png")
                    for i in range(self.wan_num_frames)
                ]
            blurs, frames, wan = self._load_group(
                [b[0] for b in burst],
                # Truncated to vjepa_window_size (an even prefix) -- see the fix note on
                # self.vjepa_window_size above; this is what actually reaches V-JEPA.
                [self._window_paths(b[3], b[1], b[2])[:self.vjepa_window_size] for b in burst],
                offset, wan_paths,
            )
            sample = {
                self.burst_key: (blurs.transpose(0, 3, 1, 2) * 2.0 - 1.0).astype(np.float32),  # [S,3,H,W]
                self.burst_frames_key: frames.transpose(0, 1, 4, 2, 3).astype(np.float32),  # [S,N,3,H,W]
            }
            if wan is not None:
                wan_chw = wan.transpose(0, 3, 1, 2).astype(np.float32)
                if self.wan_frames_range == "pm1":
                    wan_chw = wan_chw * 2.0 - 1.0
                sample[self.wan_key] = wan_chw
            if self.return_name:
                sample["name"] = f"{self._name(burst[0][0])}_burst{len(burst)}{suffix}"
            return sample

        blur_path, center, n, seq_dir, _n_seq_frames = entry
        wan_paths = None
        if self.include_wan_frames:
            wan_paths = self._wan_window_paths(seq_dir, center, self.ctx_frames or self.wan_num_frames)
        # Truncated to vjepa_window_size (an even prefix) -- see the fix note on
        # self.vjepa_window_size above; this is what actually reaches V-JEPA.
        blurs, frames, wan = self._load_group(
            [blur_path], [self._window_paths(seq_dir, center, n)[:self.vjepa_window_size]], offset, wan_paths)
        sample = {
            self.lq_key: (blurs[0].transpose(2, 0, 1) * 2.0 - 1.0).astype(np.float32),  # [3,H,W]
            self.gt_key: frames[0].transpose(0, 3, 1, 2).astype(np.float32),  # [N,3,H,W]
        }
        if wan is not None:
            wan_chw = wan.transpose(0, 3, 1, 2).astype(np.float32)
            if self.wan_frames_range == "pm1":
                wan_chw = wan_chw * 2.0 - 1.0
            if self.ctx_frames:
                s0 = (self.ctx_frames - self.wan_num_frames) // 2
                sample[self.ctx_key] = wan_chw
                sample[self.wan_key] = wan_chw[s0:s0 + self.wan_num_frames]
            else:
                sample[self.wan_key] = wan_chw
        if self.return_name:
            sample["name"] = self._name(blur_path) + suffix
        return sample

"""FallingObjectsDataset: burst-mode oracle-test adapter for the CTU FMO/TbD
"Falling Objects" dataset (`falling_imgs_gt.zip`,
https://cmp.felk.cvut.cz/fmo/) -- feeds the trained
Wan22VJEPADynamics_stageA_burst3_ar oracle checkpoint a REAL, recognizable
physical motion (an object falling under gravity) instead of Adobe240
training-distribution crops, to probe generalization/upper-bound behavior.
See findings/Wan22VJEPADynamics_findings.md's 2026-09-21 entry for the full
rationale and the input-contract derivation this mirrors.

Layout at data_dir (e.g.
/scratch/gilbreth/wweligam/data/FMO_TbD/falling_objects):
    imgs_gt/<scene>/00000000.png, 00000001.png, ...   contiguous real 240fps
                                                        sharp frames (500/scene)

Differs from Adobe240Dataset in one essential way: there are no
pre-synthesized blur files here at arbitrary center positions, so this class
SYNTHESIZES each segment's blur image itself by averaging `window_size`
consecutive sharp frames -- the exact same construction Adobe240's own blur
frames were verified to use (see adobe240_dataset.py's docstring), just
computed on the fly instead of loaded from disk. Burst grouping (non-
overlapping chunks of `burst_size` consecutive centers, `center_stride` apart)
mirrors adobe240_dataset.py's burst-mode scene discovery.

Native frames are portrait and extremely narrow (300x960, aspect ~0.31),
far outside the model's trained aspect_ratio_range ([0.5, 2.0] per
Wan22VJEPADynamics_stageA_burst3_ar.yml) -- feeding that shape directly would
confound "does the model understand this motion" with "does the model
degrade on an unseen canvas shape". Instead of squishing (non-aspect-
preserving resize) or center-cropping away most of the fall, this
pillarbox-pads each frame to square (zero-padding the narrow width) BEFORE
resizing to `out_size`, preserving the full vertical trajectory and the true
aspect ratio of the visible content while presenting the model a
square (in-distribution-shaped) canvas.
"""
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils import data as data
from torch.utils.data import default_collate


class FallingObjectsDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.data_dir = opt.data_dir
        self.sharp_subdir = opt.get("sharp_subdir", "imgs_gt")
        self.scenes_filter = opt.get("scenes", None)

        self.gt_key = opt.get("gt_key", "burst_frames")
        self.lq_key = opt.get("burst_key", "burst_blur")
        self.wan_key = opt.get("wan_key", "wan_frames")
        self.return_name = opt.get("return_name", True)

        self.window_size = int(opt.get("window_size", 11))
        self.vjepa_window_size = int(opt.get("vjepa_window_size", 10))
        if self.vjepa_window_size > self.window_size:
            raise ValueError("vjepa_window_size cannot exceed window_size")
        self.half_window = self.window_size // 2
        self.blur_window = int(opt.get("blur_window", self.window_size))
        if self.blur_window > self.window_size or (self.window_size - self.blur_window) % 2:
            raise ValueError("blur_window must be <= window_size with the same parity")
        self.blur_off = (self.window_size - self.blur_window) // 2
        # v3 context frames: when `ctx_key` is set the WHOLE loaded real window (window_size frames, e.g. 19) is also emitted under
        # that key, [window_size,3,H,W] in the wan_frames_range; wan_frames stays the centered `wan_num_frames` exposure slice.
        self.ctx_key = opt.get("ctx_key", None)

        self.center_stride = int(opt.get("center_stride", 8))
        self.margin_frames = int(opt.get("margin_frames", 20))

        self.burst_size = int(opt.get("burst_size", 3))
        if self.burst_size < 1:
            raise ValueError(f"burst_size must be >= 1, got {self.burst_size}")

        self.include_wan_frames = bool(opt.get("include_wan_frames", True))
        self.wan_num_frames = int(opt.get("wan_num_frames", 25))
        self.wan_vae_scale_factor_temporal = int(opt.get("wan_vae_scale_factor_temporal", 4))
        self.wan_prefix_frames = int(opt.get("wan_prefix_frames", 0))
        self.wan_frames_range = opt.get("wan_frames_range", "pm1")
        if self.include_wan_frames and (self.wan_num_frames + self.wan_prefix_frames - 1) % self.wan_vae_scale_factor_temporal != 0:
            raise ValueError(
                f"wan_num_frames={self.wan_num_frames} must satisfy "
                f"(wan_num_frames + wan_prefix_frames - 1) % wan_vae_scale_factor_temporal == 0"
            )

        # crop_mode="pad" (default, backward compatible with pad_to_square):
        # pillarbox-pads the native frame to square before resizing --
        # preserves the FULL frame (nothing cropped away) but leaves large
        # black margins (native frames here are 300x960, so ~65% of the
        # padded square is empty), which (a) is a canvas shape/content
        # distribution the model never saw in training and (b) shrinks the
        # object's already-small pixel footprint even further relative to
        # the frame. crop_mode="motion_square": takes a native-resolution,
        # NO-PADDING square crop (side = min(native_h, native_w), i.e. the
        # full width here) positioned via a real frame-diff motion-energy
        # scan of the burst's own span to maximize how much of the actual
        # falling-object motion lands inside the crop, then resizes that
        # crop up to out_size -- zero black padding, and the object gets a
        # larger relative footprint since we're not showing ~65% dead space.
        # Added 2026-09-21 after the padded-canvas oracle test showed the
        # model never rendering the falling object at all (see
        # findings/Wan22VJEPADynamics_findings.md) -- motion_square isolates
        # whether that failure was content (small/fast object) or partly an
        # artifact of the extreme padded canvas.
        # crop_mode="random_square" (TRAINING): a square crop of the same
        # side = min(native_h, native_w), but positioned UNIFORMLY AT RANDOM
        # per sample instead of on the motion peak. Deliberate: a
        # motion_square crop always puts a large moving object dead centre,
        # so training exclusively on those teaches the spurious rule "a crop
        # from this source always has an object in the middle" (Adobe240
        # samples, mixed in alongside, are ordinary random crops of ordinary
        # scenes). Random positioning also yields far more distinct samples
        # from the same handful of scenes, which matters because the
        # FMO/TbD half is only ~12 clips and memorizes easily. Reserve
        # motion_square for EVAL, where centring on the motion is what makes
        # the measurement discriminative.
        self.crop_mode = opt.get("crop_mode", "pad" if opt.get("pad_to_square", True) else "none")
        if self.crop_mode not in ("pad", "motion_square", "random_square", "motion_jitter", "none"):
            raise ValueError(f"Unknown crop_mode: {self.crop_mode!r}")
        self.pad_to_square = self.crop_mode == "pad"  # kept for the log line below
        self.out_size = int(opt.get("out_size", 736))

        # crop_mode="motion_jitter" (TRAINING, replaces random_square): the
        # motion-energy offset, then perturbed by +/- crop_jitter px.
        #
        # WHY: random_square draws the crop offset INDEPENDENTLY of which burst
        # was selected -- `top = random.randint(0, H - side)` is uniform over
        # 0..660 on FMO's 300x960 portrait frames, while the object occupies a
        # specific ~250px band. Measured over all 796 FMO bursts x all 661
        # offsets (gt_bbox, 2026-09-23): the object is visible in ZERO frames in
        # 58.0% of samples, in >=50% of frames in only 28.6%, and the full
        # trajectory appears in only 15.1%. FMO is the entire reason this
        # dataset exists, and it was supplying mostly-empty background.
        #
        # motion_square alone fixes presence but centres the object in every
        # sample, removing positional diversity (the model can then learn
        # "the object is always at the centre"). The jitter restores diversity
        # while keeping the object in frame. Per-burst vertical travel is
        # 222-290px against a 300px window, so the usable slack is 10-78px --
        # keep crop_jitter well under that (24 is the configured default here);
        # a larger value trades trajectory coverage for position diversity.
        self.crop_jitter = int(opt.get("crop_jitter", 24))
        if self.crop_jitter < 0:
            raise ValueError(f"crop_jitter must be >= 0, got {self.crop_jitter}")

        self.max_bursts_per_scene = opt.get("max_bursts_per_scene", None)
        if self.max_bursts_per_scene is not None:
            self.max_bursts_per_scene = int(self.max_bursts_per_scene)

        # Optional {scene: peak_motion_frame_idx} override -- when given,
        # builds exactly ONE burst per listed scene, centered on that frame
        # (centers = [peak-stride, peak, peak+stride]) instead of the
        # generic evenly-spaced-across-the-clip selection below. Use this to
        # deliberately land a burst's window on the scene's actual
        # high-motion interval (found e.g. via frame-to-frame pixel diff)
        # rather than risk an evenly-spaced pick landing on a mostly-static
        # pre-drop/post-landing stretch, which happened for several scenes
        # in this project's first pass -- see
        # findings/Wan22VJEPADynamics_findings.md's 2026-09-21 entry.
        # NOTE: kept as whatever mapping-like object opt.get() returns (an
        # OmegaConf DictConfig here, but possibly this repo's DictAsMember
        # elsewhere) and accessed ONLY via .get() below -- DictAsMember
        # (utils/misc.py) subclasses `dict` but stores its actual contents in
        # `self.__dict__` rather than the real dict storage, so `dict(x)`,
        # `key in x`, and `x[key]` on it silently operate on an EMPTY dict
        # (no error, just always "not found") while `x.get(key, default)` is
        # the one method it correctly overrides -- verified the hard way via
        # a real crash (dict(peak_centers) produced an empty dict, so every
        # scene appeared missing and __init__ raised "No valid bursts
        # constructed"), see findings/Wan22VJEPADynamics_findings.md's
        # 2026-09-21 entry.
        self.peak_center_scenes = opt.get("peak_center_scenes", None)

        # centers_around_peak=N (TRAINING): instead of ONE burst centred on
        # the peak, build every burst whose centers lie within peak+/-N, so a
        # scene contributes many samples concentrated on the interval where
        # the object is actually moving. Without this, bursts spread over the
        # whole clip are mostly pre-drop/post-landing static stretches and the
        # localized-motion signal gets diluted to a few percent of the mix.
        # burst_center_step: how far (in centers) consecutive bursts advance;
        # default burst_size = non-overlapping (the eval/Adobe240 convention),
        # set to 1 for maximally-overlapping training bursts. Note this only
        # changes which bursts exist, never the WITHIN-burst spacing, which
        # stays center_stride so each burst keeps the temporal structure the
        # model was trained/evaluated with.
        self.centers_around_peak = opt.get("centers_around_peak", None)
        if self.centers_around_peak is not None:
            self.centers_around_peak = int(self.centers_around_peak)
        self.burst_center_step = int(opt.get("burst_center_step", self.burst_size))
        if self.burst_center_step < 1:
            raise ValueError(f"burst_center_step must be >= 1, got {self.burst_center_step}")

        extensions = tuple(opt.get("extensions", [".png", ".jpg", ".jpeg"]))

        sharp_root = os.path.join(self.data_dir, self.sharp_subdir)
        scenes = sorted(
            d for d in os.listdir(sharp_root)
            if os.path.isdir(os.path.join(sharp_root, d))
            and (self.scenes_filter is None or d in self.scenes_filter)
        )
        if not scenes:
            raise ValueError(f"No scenes found under {sharp_root} (scenes filter={self.scenes_filter!r})")

        # self.samples: list of (scene_dir, ext, centers) -- one entry per burst.
        self.samples: List[Tuple[str, str, List[int]]] = []
        for scene in scenes:
            scene_dir = os.path.join(sharp_root, scene)
            names = sorted(
                f for f in os.listdir(scene_dir) if f.lower().endswith(extensions)
            )
            if not names:
                continue
            ext = os.path.splitext(names[0])[1]
            indices = sorted(int(os.path.splitext(n)[0]) for n in names)
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"Non-contiguous frame indices in {scene_dir}")
            lo = indices[0] + self.half_window + self.margin_frames
            hi = indices[-1] - self.half_window - self.margin_frames

            if self.peak_center_scenes is not None:
                peak = self.peak_center_scenes.get(scene, None)
                if peak is None:
                    continue
                peak = int(peak)

                if self.centers_around_peak is not None:
                    # TRAINING: many bursts spanning peak +/- centers_around_peak.
                    span_lo = max(lo, peak - self.centers_around_peak)
                    span_hi = min(hi, peak + self.centers_around_peak)
                    cands = list(range(span_lo, span_hi + 1, self.center_stride))
                    added = 0
                    for i in range(0, len(cands) - self.burst_size + 1, self.burst_center_step):
                        self.samples.append((scene_dir, ext, cands[i:i + self.burst_size]))
                        added += 1
                    if added == 0:
                        raise ValueError(
                            f"centers_around_peak={self.centers_around_peak} around "
                            f"peak={peak} yields no full burst for {scene} "
                            f"(valid center range [{lo},{hi}], burst_size={self.burst_size})"
                        )
                    continue

                # EVAL: exactly one burst centred on the peak.
                mid_offset = (self.burst_size - 1) // 2
                first_center = peak - mid_offset * self.center_stride
                centers = [first_center + i * self.center_stride for i in range(self.burst_size)]
                if centers[0] < lo or centers[-1] > hi:
                    raise ValueError(
                        f"peak_center_scenes[{scene}]={peak} produces centers {centers} outside "
                        f"the valid range [{lo}, {hi}] for this scene -- reduce center_stride/burst_size "
                        f"or pick a different peak."
                    )
                self.samples.append((scene_dir, ext, centers))
                continue

            centers_all = list(range(lo, hi + 1, self.center_stride))

            bursts = [
                centers_all[i:i + self.burst_size]
                for i in range(0, len(centers_all) - self.burst_size + 1, self.burst_size)
            ]
            if self.max_bursts_per_scene is not None and len(bursts) > self.max_bursts_per_scene:
                # Evenly spaced across the WHOLE clip (not just the first N),
                # since the actual drop/motion can occur anywhere in the
                # scene's timeline (frequently well past the first few
                # bursts -- verified empirically for v_box_GTgamma, where
                # motion peaks around frame ~280-305 out of 500).
                pick_idx = np.linspace(0, len(bursts) - 1, self.max_bursts_per_scene)
                bursts = [bursts[int(round(i))] for i in pick_idx]
            for centers in bursts:
                self.samples.append((scene_dir, ext, centers))

        if not self.samples:
            raise ValueError(f"No valid bursts constructed from {sharp_root}")
        print(
            f"Initialized FallingObjectsDataset with {len(self.samples)} bursts across "
            f"{len(scenes)} scenes (window_size={self.window_size}, "
            f"vjepa_window_size={self.vjepa_window_size}, burst_size={self.burst_size}, "
            f"out_size={self.out_size}, pad_to_square={self.pad_to_square})"
        )

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        return default_collate(batch)

    @staticmethod
    def _load01(path: str) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def _resize01(self, img01: np.ndarray) -> np.ndarray:
        img = Image.fromarray((img01 * 255.0).astype(np.uint8)).resize((self.out_size, self.out_size))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _pad_to_square01(self, img01: np.ndarray) -> np.ndarray:
        h, w = img01.shape[:2]
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=np.float32)
        top = (side - h) // 2
        left = (side - w) // 2
        canvas[top:top + h, left:left + w] = img01
        return canvas

    @staticmethod
    def _motion_square_offset(raw01_stack: np.ndarray, side: int) -> Tuple[int, int]:
        """raw01_stack: [T,H,W,3] float32 in [0,1] (native resolution, this
        burst's full span). Finds the (top, left) offset of the `side` x
        `side` window that captures the most total frame-to-frame motion
        energy (sum of abs grayscale diffs) -- only the longer of H/W is
        searched over (the shorter one already fits `side` exactly, since
        side = min(H, W))."""
        gray = raw01_stack.mean(axis=3)  # [T,H,W]
        diffs = np.abs(np.diff(gray, axis=0)).sum(axis=0)  # [H,W]
        H, W = diffs.shape
        if H >= W:
            row_energy = diffs.sum(axis=1)  # [H]
            csum = np.cumsum(np.concatenate([[0.0], row_energy]))
            sums = csum[side:] - csum[:-side]
            top = int(np.argmax(sums))
            return top, 0
        else:
            col_energy = diffs.sum(axis=0)  # [W]
            csum = np.cumsum(np.concatenate([[0.0], col_energy]))
            sums = csum[side:] - csum[:-side]
            left = int(np.argmax(sums))
            return 0, left

    def __getitem__(self, idx):
        scene_dir, ext, centers = self.samples[idx]
        full_span_start = centers[0] - self.half_window
        full_span_end = centers[-1] + self.half_window
        full_frame_paths = [
            os.path.join(scene_dir, f"{i:08d}{ext}")
            for i in range(full_span_start, full_span_end + 1)
        ]

        if self.crop_mode in ("motion_square", "random_square", "motion_jitter"):
            raw01_stack = np.stack([self._load01(p) for p in full_frame_paths])  # [span,H,W,3]
            H, W = raw01_stack.shape[1:3]
            side = min(H, W)
            if self.crop_mode == "motion_square":
                top, left = self._motion_square_offset(raw01_stack, side)
            elif self.crop_mode == "motion_jitter":
                top, left = self._motion_square_offset(raw01_stack, side)
                if self.crop_jitter > 0:
                    j = self.crop_jitter
                    # Clipping makes this a no-op on whichever axis already fits
                    # exactly (side == min(H, W), so that axis has range 0) --
                    # FMO jitters vertically, TbD (960x400) horizontally.
                    top = int(np.clip(top + random.randint(-j, j), 0, H - side))
                    left = int(np.clip(left + random.randint(-j, j), 0, W - side))
            else:
                top = random.randint(0, H - side)
                left = random.randint(0, W - side)
            cropped = raw01_stack[:, top:top + side, left:left + side]  # [span,side,side,3]
            full_frames01 = np.stack([self._resize01(f) for f in cropped])
        elif self.crop_mode == "pad":
            full_frames01 = np.stack([
                self._resize01(self._pad_to_square01(self._load01(p))) for p in full_frame_paths
            ])
        else:  # "none" -- straight resize, only sane when native aspect is already in-range
            full_frames01 = np.stack([self._resize01(self._load01(p)) for p in full_frame_paths])

        burst_blur01 = []
        burst_frames = []
        for c in centers:
            seg_start = (c - self.half_window) - full_span_start
            seg_window = full_frames01[seg_start:seg_start + self.window_size]  # [window_size,H,W,3]
            burst_blur01.append(seg_window[self.blur_off:self.blur_off + self.blur_window].mean(axis=0))
            burst_frames.append(seg_window[: self.vjepa_window_size])  # even prefix for V-JEPA
        burst_blur01 = np.stack(burst_blur01)  # [B,H,W,3]
        burst_frames = np.stack(burst_frames)  # [B,vjepa_window_size,H,W,3]

        burst_blur_pm1 = (burst_blur01.transpose(0, 3, 1, 2) * 2.0 - 1.0).astype(np.float32)  # [B,3,H,W]
        burst_frames = burst_frames.transpose(0, 1, 4, 2, 3).astype(np.float32)  # [B,T,3,H,W]

        if self.burst_size == 1:
            # SINGLE-FRAME mode: Wan22VJEPADynamics_model's burst_size==1 path
            # indexes hardcoded "blur"/"wan_frames" keys (NOT burst_blur/
            # burst_frames -- see its _dynamics_full_grid vs
            # _dynamics_full_grid_burst dispatch), and its oracle vjepa_gt
            # branch reads the V-JEPA clip straight out of `wan_frames`
            # rather than a separate per-segment frames key. Mirrors
            # Adobe240Dataset._finalize_sample's single-frame output exactly.
            sample = {
                "blur": burst_blur_pm1[0],                     # [3,H,W] in [-1,1]
                "frames": burst_frames[0],                     # [T,3,H,W] in [0,1], parity only
            }
        else:
            sample = {
                self.lq_key: burst_blur_pm1,
                self.gt_key: burst_frames,
            }

        if self.include_wan_frames:
            full_span_len = full_frames01.shape[0]
            wan_start = (full_span_len - self.wan_num_frames) // 2
            if wan_start < 0:
                raise ValueError(
                    f"wan_num_frames={self.wan_num_frames} exceeds the full burst span "
                    f"({full_span_len} frames)"
                )
            wan_frames01 = full_frames01[wan_start:wan_start + self.wan_num_frames]
            wan_frames01 = wan_frames01.transpose(0, 3, 1, 2).astype(np.float32)  # [T,3,H,W]
            if self.wan_frames_range == "pm1":
                wan_frames01 = wan_frames01 * 2.0 - 1.0
            sample[self.wan_key] = wan_frames01
            if self.ctx_key and self.burst_size == 1:
                ctx01 = full_frames01.transpose(0, 3, 1, 2).astype(np.float32)
                sample[self.ctx_key] = ctx01 * 2.0 - 1.0 if self.wan_frames_range == "pm1" else ctx01

        if self.return_name:
            scene = os.path.basename(scene_dir)
            suffix = "single" if self.burst_size == 1 else f"burst{len(centers)}"
            sample["name"] = f"{scene}_c{centers[0]}_{suffix}"
        return sample

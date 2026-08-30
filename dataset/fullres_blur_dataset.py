"""FullResBlurDataset: JEPA-style block replacement for training a
fidelity-preserving (VAE) encoder to be blur-robust.

Loads the existing full-resolution GT/blur pairs from disk (Deblur_MIOR
already ships both) rather than synthesizing blur on the fly -- no kernel
generation, no cv2.filter2D.

This mimics how JEPA actually trains, not "encode the whole blurred image and
match it everywhere": most of the frame stays the clean GT (context, visible
to the encoder exactly as-is), and one rectangular block is replaced by the
corresponding region of the real blurred image (the corrupted "target"
region) -- blur substitutes for JEPA's mask token, since the corrupted region
here is degraded rather than removed, per the original request. The loss (in
the model) is computed ONLY at the replaced positions, using the surrounding
clean context the encoder can see directly -- exactly analogous to JEPA
predicting masked-patch embeddings from visible-patch context, rather than
"deblur the whole frame with no clean reference to lean on."

The CROP taken from each full-res pair also has a randomized aspect ratio
(area held roughly fixed at crop_size^2, so per-sample compute stays
comparable) -- separate from, and unrelated to, the replaced block's own
aspect ratio below. This is so the VAE encoder sees varied input shapes
during training rather than only ever square ones, since it needs to handle
arbitrary aspect ratios at inference. Because samples in a batch can now have
different (H,W), train configs using this dataset must use batch_size: 1
(with gradient_accumulation_steps to make up any effective-batch-size
difference) -- the default collate_fn cannot stack tensors of different shape.

Returns, per sample (shape varies sample-to-sample, fixed within one sample):
  gt_key   -> the clean crop, unmodified                         [3,H,W]
  lq_key   -> the crop with one block replaced by blur            [3,H,W]
  mask_key -> 1.0 inside the replaced block, 0.0 elsewhere        [1,H,W]
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils import data as data


class FullResBlurDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.data_dir = opt.data_dir
        self.gt_folder = opt.get("gt_folder", "gt")
        self.lq_folder = opt.get("lq_folder", "blur")
        self.gt_key = opt.get("gt_key", "gt")
        self.lq_key = opt.get("lq_key", "blur")
        self.mask_key = opt.get("mask_key", "mask")
        self.is_train = opt.get("is_train", True)
        self.return_name = opt.get("return_name", False)

        # Crop AREA is held roughly fixed at crop_size^2; the crop's shape
        # (H,W) is randomized around that area via crop_aspect_min/max, so the
        # VAE sees varied aspect ratios rather than only square inputs. This
        # is independent of block_aspect_min/max below, which shapes the
        # replaced sub-patch *within* whatever crop shape gets drawn.
        self.crop_size = opt.get("crop_size", 512)
        self.crop_area = self.crop_size * self.crop_size
        self.crop_aspect_min = opt.get("crop_aspect_min", 0.5)
        self.crop_aspect_max = opt.get("crop_aspect_max", 2.0)
        # Crop H/W must be a multiple of this for the VAE's patchify/pack path
        # (needs an even latent grid); 32 is a safe multiple of the actual
        # ~16px spatial divisor this VAE config uses, without hardcoding it.
        self.shape_divisor = opt.get("shape_divisor", 32)

        # Fraction of the crop's area covered by the single replaced block --
        # JEPA/I-JEPA-style block masking typically covers a large minority-
        # to-majority share of the frame (I-JEPA's target blocks: ~15-20% each,
        # several of them; single-block MAE-style setups often go to ~50-75%).
        # Default to a moderate single block so there's always a comparable
        # amount of clean context left to condition on.
        self.replace_ratio_min = opt.get("replace_ratio_min", 0.3)
        self.replace_ratio_max = opt.get("replace_ratio_max", 0.5)
        self.block_aspect_min = opt.get("block_aspect_min", 0.5)
        self.block_aspect_max = opt.get("block_aspect_max", 2.0)

        extensions = tuple(opt.get("extensions", [".png", ".jpg", ".jpeg", ".bmp", ".webp"]))
        gt_dir = os.path.join(self.data_dir, self.gt_folder)
        lq_dir = os.path.join(self.data_dir, self.lq_folder)
        gt_names = sorted(
            f for f in os.listdir(gt_dir) if f.lower().endswith(extensions)
        )
        lq_names = set(os.listdir(lq_dir))
        names = [f for f in gt_names if f in lq_names]
        if len(names) != len(gt_names):
            print(f"[warn] {len(gt_names) - len(names)} GT images have no matching blur file.")
        if not names:
            raise ValueError(f"No matching GT/blur pairs under {self.data_dir} "
                              f"({self.gt_folder}/ vs {self.lq_folder}/)")

        split_range = opt.get("split_range", None)
        if split_range is not None:
            lo = int(split_range[0] / 100 * len(names))
            hi = int(split_range[1] / 100 * len(names))
            names = names[lo:hi]
        self.names = names
        self.gt_dir = gt_dir
        self.lq_dir = lq_dir

        # Val draws a fixed crop shape and a fixed replaced block per index
        # (built lazily, cached) so repeated validation passes across
        # training steps see the same crop geometry and degradation --
        # otherwise "did the model get better" would be confounded with "did
        # this epoch's random draw happen to be easier."
        self.crop_shape_pool = []
        self.block_pool = []

        print(
            f"Initialized FullResBlurDataset with {len(self.names)} GT/blur pairs "
            f"from {self.data_dir} (is_train={self.is_train}, crop_area~{self.crop_size}^2, "
            f"crop_aspect=[{self.crop_aspect_min},{self.crop_aspect_max}], "
            f"replace_ratio=[{self.replace_ratio_min},{self.replace_ratio_max}])"
        )

    def __len__(self):
        return len(self.names)

    def _round_to_divisor(self, x, max_x):
        x = (min(x, max_x) // self.shape_divisor) * self.shape_divisor
        return max(self.shape_divisor, x)

    def _sample_crop_shape(self, h, w):
        """Random (ch, cw) with area ~= crop_size^2 and a random aspect ratio,
        rounded to shape_divisor and clipped to fit inside the source image."""
        aspect = random.uniform(self.crop_aspect_min, self.crop_aspect_max)
        ch = int(round((self.crop_area * aspect) ** 0.5))
        cw = int(round((self.crop_area / aspect) ** 0.5))
        return self._round_to_divisor(ch, h), self._round_to_divisor(cw, w)

    def _crop_shape_for_idx(self, idx, h, w):
        if self.is_train:
            return self._sample_crop_shape(h, w)
        while len(self.crop_shape_pool) <= idx:
            self.crop_shape_pool.append(self._sample_crop_shape(h, w))
        return self.crop_shape_pool[idx]

    def _sample_block(self, ch, cw):
        """One rectangular block (top, left, bh, bw) covering a random area
        fraction of the crop, random aspect ratio, random position."""
        if self.replace_ratio_min >= 1.0 and self.replace_ratio_max >= 1.0:
            # "Cover everything" must mean the literal (ch, cw) rectangle,
            # not a derived-aspect-ratio square -- with non-square crops now
            # in play, aspect=1.0 no longer coincides with "the whole crop"
            # (it'd be sqrt(ch*cw) on a side, leaving GT strips on one edge).
            # This is what the full-frame/no-patching baseline relies on.
            return 0, 0, ch, cw
        area = ch * cw
        target_area = random.uniform(self.replace_ratio_min, self.replace_ratio_max) * area
        aspect = random.uniform(self.block_aspect_min, self.block_aspect_max)
        bh = min(ch, max(1, int(round((target_area * aspect) ** 0.5))))
        bw = min(cw, max(1, int(round((target_area / aspect) ** 0.5))))
        top = random.randint(0, ch - bh)
        left = random.randint(0, cw - bw)
        return top, left, bh, bw

    def _sample_block_for_idx(self, idx, ch, cw):
        if self.is_train:
            return self._sample_block(ch, cw)
        while len(self.block_pool) <= idx:
            self.block_pool.append(self._sample_block(ch, cw))
        return self.block_pool[idx]

    def __getitem__(self, idx):
        name = self.names[idx]
        gt_img = Image.open(os.path.join(self.gt_dir, name)).convert("RGB")
        blur_img = Image.open(os.path.join(self.lq_dir, name)).convert("RGB")
        gt_arr = np.array(gt_img).astype(np.float32)  # [H,W,3] in [0,255]
        blur_arr = np.array(blur_img).astype(np.float32)
        h, w = gt_arr.shape[:2]

        # Reference size = the largest crop side this dataset could ever draw
        # (widest aspect ratio at the target area) -- upsample both images if
        # the source is smaller than that, rather than crashing mid-epoch.
        max_side = int(round((self.crop_area * self.crop_aspect_max) ** 0.5)) + self.shape_divisor
        if h < max_side or w < max_side:
            scale = max_side / min(h, w) * 1.05
            new_size = (int(w * scale) + 1, int(h * scale) + 1)
            gt_arr = np.array(gt_img.resize(new_size, Image.BICUBIC)).astype(np.float32)
            blur_arr = np.array(blur_img.resize(new_size, Image.BICUBIC)).astype(np.float32)
            h, w = gt_arr.shape[:2]

        ch, cw = self._crop_shape_for_idx(idx, h, w)

        if self.is_train:
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
        else:
            top, left = (h - ch) // 2, (w - cw) // 2

        gt_crop = gt_arr[top:top + ch, left:left + cw]
        blur_crop = blur_arr[top:top + ch, left:left + cw]

        # Splice: mixed = GT everywhere except one block, which is replaced by
        # the real blurred image's pixels there -- the actual "blur/gt patches
        # in the same input" mechanism, as opposed to two separate full-blur
        # and full-gt images.
        b_top, b_left, bh, bw = self._sample_block_for_idx(idx, ch, cw)
        mask = np.zeros((ch, cw), dtype=np.float32)
        mask[b_top:b_top + bh, b_left:b_left + bw] = 1.0

        mixed_crop = gt_crop.copy()
        mixed_crop[b_top:b_top + bh, b_left:b_left + bw] = blur_crop[b_top:b_top + bh, b_left:b_left + bw]

        gt_t = torch.from_numpy((gt_crop.transpose(2, 0, 1) / 127.5 - 1.0).copy()).float()
        mixed_t = torch.from_numpy((mixed_crop.transpose(2, 0, 1) / 127.5 - 1.0).copy()).float()
        mask_t = torch.from_numpy(mask[None].copy()).float()

        out = {self.gt_key: gt_t, self.lq_key: mixed_t, self.mask_key: mask_t}
        if self.return_name:
            out["name"] = name
        return out

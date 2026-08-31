"""FullResPatchifierDataset: same JEPA-style block-replacement sample as
FullResBlurDataset, but yields REAL stacked batches instead of batch_size=1.

FullResBlurDataset draws an independent random crop aspect ratio per sample,
so two samples in a batch can land on different (H,W) -- the default
collate_fn's torch.stack then crashes ("stack expects each tensor to be equal
size"). That crash is real and was hit in practice once val.batch_size > 1
actually pulled two differently-shaped crops into one batch.

The fix here is a producer pattern, not padding: load `batch_size` full-res
GT/blur pairs from disk ONCE, then draw `patches_per_group` different crops
from that same in-memory group before moving on to the next `batch_size`
images. Every crop within one yielded batch shares ONE randomly-drawn shape
(so it can actually be torch.stack'd), while the block position/size (the
JEPA-style replaced region) is still drawn independently per image in the
group, and the shape itself is redrawn fresh for every one of the
`patches_per_group` yields -- so aspect-ratio diversity is preserved across
yields, just not *within* a single stacked batch.

This is an IterableDataset -- __iter__ yields already-batched dicts of shape
[batch_size, C, H, W], so the DataLoader must be built with batch_size=None
(see VAEBlurAlign_model.setup_dataloaders). Reusing each loaded image for
`patches_per_group` crops instead of 1 amortizes the disk read + JPEG/PNG
decode cost, which otherwise dominates at these crop sizes.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils import data as data


class FullResPatchifierDataset(data.IterableDataset):
    def __init__(self, opt):
        super().__init__()
        self.data_dir = opt.data_dir
        self.gt_folder = opt.get("gt_folder", "gt")
        self.lq_folder = opt.get("lq_folder", "blur")
        self.gt_key = opt.get("gt_key", "gt")
        self.lq_key = opt.get("lq_key", "blur")
        self.mask_key = opt.get("mask_key", "mask")
        self.is_train = opt.get("is_train", True)

        self.batch_size = int(opt.get("batch_size", 4))
        # How many different crops to draw from one loaded group of
        # `batch_size` full-res images before reloading a fresh group from
        # disk -- the actual disk-I/O amortization knob.
        self.patches_per_group = int(opt.get("patches_per_group", 10))

        self.crop_size = opt.get("crop_size", 512)
        self.crop_area = self.crop_size * self.crop_size
        self.crop_aspect_min = opt.get("crop_aspect_min", 0.5)
        self.crop_aspect_max = opt.get("crop_aspect_max", 2.0)
        self.shape_divisor = opt.get("shape_divisor", 32)

        self.replace_ratio_min = opt.get("replace_ratio_min", 0.3)
        self.replace_ratio_max = opt.get("replace_ratio_max", 0.5)
        self.block_aspect_min = opt.get("block_aspect_min", 0.5)
        self.block_aspect_max = opt.get("block_aspect_max", 2.0)

        # Reject block placements whose own blur-vs-GT signal is too weak to
        # ever be recoverable: an overfit-per-sample sweep found the frozen
        # VAE's own decode(encode(x)) round trip already costs ~0.0005-0.0019
        # MSE (in the same [-1,1] pixel scale used everywhere in training) on
        # a CLEAN image with zero adaptation -- for any block whose true
        # blur-vs-GT MSE is below that, no trained encoder can ever beat the
        # "do nothing" floor, because the round trip alone injects more error
        # than the blur did. ~11/20 randomly-placed blocks landed below this
        # in practice, which is why population-level training always looked
        # worse than doing nothing: a large fraction of every batch was an
        # unwinnable comparison by construction, not a training failure.
        self.min_block_floor_mse = opt.get("min_block_floor_mse", 0.002)
        self.max_block_resample_attempts = int(opt.get("max_block_resample_attempts", 20))

        # Largest side any sampled crop shape could ever need -- every loaded
        # image gets upsampled to at least this on both sides, which
        # guarantees any later-sampled shared (ch, cw) fits every image in a
        # group regardless of the images' differing native resolutions.
        self.max_side = int(round((self.crop_area * self.crop_aspect_max) ** 0.5)) + self.shape_divisor

        extensions = tuple(opt.get("extensions", [".png", ".jpg", ".jpeg", ".bmp", ".webp"]))
        gt_dir = os.path.join(self.data_dir, self.gt_folder)
        lq_dir = os.path.join(self.data_dir, self.lq_folder)
        gt_names = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(extensions))
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

        n_groups = max(1, len(self.names) // self.batch_size)
        print(
            f"Initialized FullResPatchifierDataset with {len(self.names)} GT/blur pairs "
            f"from {self.data_dir} (is_train={self.is_train}, batch_size={self.batch_size}, "
            f"patches_per_group={self.patches_per_group}, ~{n_groups} groups/epoch, "
            f"crop_area~{self.crop_size}^2, crop_aspect=[{self.crop_aspect_min},{self.crop_aspect_max}], "
            f"replace_ratio=[{self.replace_ratio_min},{self.replace_ratio_max}])"
        )

    def __len__(self):
        n_groups = max(1, len(self.names) // self.batch_size)
        return n_groups * self.patches_per_group

    def _round_to_divisor(self, x, max_x):
        x = (min(x, max_x) // self.shape_divisor) * self.shape_divisor
        return max(self.shape_divisor, x)

    def _sample_crop_shape(self, rng):
        aspect = rng.uniform(self.crop_aspect_min, self.crop_aspect_max)
        ch = int(round((self.crop_area * aspect) ** 0.5))
        cw = int(round((self.crop_area / aspect) ** 0.5))
        return self._round_to_divisor(ch, self.max_side), self._round_to_divisor(cw, self.max_side)

    def _sample_block(self, rng, gt_crop, blur_crop, ch, cw):
        if self.replace_ratio_min >= 1.0 and self.replace_ratio_max >= 1.0:
            return 0, 0, ch, cw

        best = None  # (mse, top, left, bh, bw) -- best-effort fallback
        for _attempt in range(self.max_block_resample_attempts):
            area = ch * cw
            target_area = rng.uniform(self.replace_ratio_min, self.replace_ratio_max) * area
            aspect = rng.uniform(self.block_aspect_min, self.block_aspect_max)
            bh = min(ch, max(1, int(round((target_area * aspect) ** 0.5))))
            bw = min(cw, max(1, int(round((target_area / aspect) ** 0.5))))
            top = rng.randint(0, ch - bh)
            left = rng.randint(0, cw - bw)

            # gt_crop/blur_crop are raw [0,255] arrays -- compare on the same
            # [-1,1] scale used everywhere else in training so this threshold
            # is directly comparable to the measured VAE round-trip floor.
            gt_block = gt_crop[top:top + bh, left:left + bw] / 127.5 - 1.0
            blur_block = blur_crop[top:top + bh, left:left + bw] / 127.5 - 1.0
            mse = float(np.mean((gt_block - blur_block) ** 2))

            if best is None or mse > best[0]:
                best = (mse, top, left, bh, bw)
            if mse >= self.min_block_floor_mse:
                return top, left, bh, bw

        # Nothing cleared the threshold in max_block_resample_attempts tries
        # (this crop may just not have a strongly-blurred region) -- use
        # whatever had the strongest signal rather than looping forever.
        return best[1], best[2], best[3], best[4]

    def _load_pair(self, name):
        gt_img = Image.open(os.path.join(self.gt_dir, name)).convert("RGB")
        blur_img = Image.open(os.path.join(self.lq_dir, name)).convert("RGB")
        gt_arr = np.array(gt_img).astype(np.float32)
        blur_arr = np.array(blur_img).astype(np.float32)
        h, w = gt_arr.shape[:2]
        if h < self.max_side or w < self.max_side:
            scale = self.max_side / min(h, w) * 1.05
            new_size = (int(w * scale) + 1, int(h * scale) + 1)
            gt_arr = np.array(gt_img.resize(new_size, Image.BICUBIC)).astype(np.float32)
            blur_arr = np.array(blur_img.resize(new_size, Image.BICUBIC)).astype(np.float32)
        # Per-pixel blur energy over the FULL image, as an integral image
        # (computed ONCE per loaded pair, reused for every one of the
        # patches_per_group crop draws below) -- used to bias crop placement
        # toward where blur actually is (see _pick_crop_position). Motion
        # blur is typically localized (a moving subject against a sharp
        # background), so a uniformly-random crop position often misses it
        # entirely; retrying the BLOCK position within an already-blur-free
        # crop can't fix that -- confirmed empirically: for several crops,
        # the best block found by exhaustively searching the whole crop was
        # still below min_block_floor_mse, meaning the crop itself had ~no
        # blur anywhere, not that block placement was unlucky.
        diff_energy = ((gt_arr - blur_arr) ** 2).mean(axis=-1)
        eh, ew = diff_energy.shape
        integral = np.zeros((eh + 1, ew + 1), dtype=np.float64)
        np.cumsum(diff_energy, axis=0, out=integral[1:, 1:])
        np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])
        return gt_arr, blur_arr, integral

    @staticmethod
    def _pick_crop_position(rng, integral, ch, cw):
        h, w = integral.shape[0] - 1, integral.shape[1] - 1
        n_top, n_left = h - ch + 1, w - cw + 1
        if n_top <= 1 and n_left <= 1:
            return 0, 0
        # O(1) sum over any ch x cw window via the precomputed integral image
        # -- scoring EVERY valid crop position is one vectorized slice, no
        # per-draw cumsum needed.
        window_sum = (
            integral[ch:ch + n_top, cw:cw + n_left]
            - integral[0:n_top, cw:cw + n_left]
            - integral[ch:ch + n_top, 0:n_left]
            + integral[0:n_top, 0:n_left]
        )
        # Sample among the highest-energy positions (not a pure argmax) so
        # crop placement still varies across draws instead of collapsing onto
        # one fixed location every time.
        flat = window_sum.reshape(-1)
        k = min(20, flat.size)
        top_k_idx = np.argpartition(flat, -k)[-k:]
        chosen = top_k_idx[rng.randrange(len(top_k_idx))]
        return int(chosen // n_left), int(chosen % n_left)

    @staticmethod
    def _to_tensor(arr_hwc):
        return torch.from_numpy((arr_hwc.transpose(2, 0, 1) / 127.5 - 1.0).copy()).float()

    def __iter__(self):
        # Validation must draw the exact same sequence of groups/crops/blocks
        # every time __iter__ runs (each call to validation() re-iterates the
        # dataloader from scratch) -- otherwise "did the model get better"
        # would be confounded with "did this pass happen to draw an easier
        # crop." A fresh, fixed-seed RNG gives that determinism without
        # needing FullResBlurDataset's old per-index cache.
        worker_info = data.get_worker_info()
        if self.is_train:
            # Python's global `random` module is NOT auto-reseeded across
            # forked DataLoader workers (unlike torch's own RNG) -- without
            # this, every worker would inherit the identical post-fork state
            # and draw the exact same sequence of crop shapes/blocks, just
            # applied to different images via the index sharding below.
            rng = random.Random(worker_info.seed) if worker_info is not None else random
        else:
            rng = random.Random(1234)

        order = list(range(len(self.names)))
        if self.is_train:
            rng.shuffle(order)
        if worker_info is not None:
            # Disjoint index shard per worker -- IterableDataset has no
            # built-in batching/sharding, so without this every worker would
            # independently iterate the FULL image list and every group would
            # get yielded once per worker.
            order = order[worker_info.id::worker_info.num_workers]

        n_groups = len(order) // self.batch_size
        for g in range(n_groups):
            group_idx = order[g * self.batch_size:(g + 1) * self.batch_size]
            loaded = [self._load_pair(self.names[i]) for i in group_idx]

            for _ in range(self.patches_per_group):
                ch, cw = self._sample_crop_shape(rng)
                gt_list, mixed_list, mask_list = [], [], []
                for gt_arr, blur_arr, diff_integral in loaded:
                    top, left = self._pick_crop_position(rng, diff_integral, ch, cw)
                    gt_crop = gt_arr[top:top + ch, left:left + cw]
                    blur_crop = blur_arr[top:top + ch, left:left + cw]

                    b_top, b_left, bh, bw = self._sample_block(rng, gt_crop, blur_crop, ch, cw)
                    mask = np.zeros((ch, cw), dtype=np.float32)
                    mask[b_top:b_top + bh, b_left:b_left + bw] = 1.0

                    mixed_crop = gt_crop.copy()
                    mixed_crop[b_top:b_top + bh, b_left:b_left + bw] = \
                        blur_crop[b_top:b_top + bh, b_left:b_left + bw]

                    gt_list.append(self._to_tensor(gt_crop))
                    mixed_list.append(self._to_tensor(mixed_crop))
                    mask_list.append(torch.from_numpy(mask[None].copy()).float())

                yield {
                    self.gt_key: torch.stack(gt_list),
                    self.lq_key: torch.stack(mixed_list),
                    self.mask_key: torch.stack(mask_list),
                }

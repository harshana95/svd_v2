"""MixedDataset: draws samples from several sub-datasets at a fixed, explicit
ratio, so a fine-tune can teach a new capability on a small dataset while
continuing to see the original training distribution and avoid catastrophic
forgetting.

Built for the Wan22VJEPADynamics fine-tune that adds localized falling-object
motion (FMO/TbD) on top of Adobe240 camera-shake motion -- see
findings/Wan22VJEPADynamics_findings.md's 2026-09-21 entries for why those two
are structurally different tasks and why the model trained on Adobe240 alone
does not render the former.

Config:
    type: MixedDataset
    epoch_size: 4000            # samples per epoch (indices this reports)
    seed: 42                    # shuffles the source plan deterministically
    sources:
      - weight: 0.75            # fraction of samples drawn from this source
        type: Adobe240Dataset
        ...that dataset's own options...
      - weight: 0.25
        type: FallingObjectsDataset
        ...

DEFAULT MODE (patch_aspect_ratios unset/false) -- every source must emit the
SAME dict keys and the SAME tensor shapes, because samples from different
sources land in the same batch and `default_collate` stacks them. In practice
that means giving every source the same square output size (Adobe240Dataset
`crop_size: N` with NO `aspect_ratio_range`; FallingObjectsDataset
`crop_mode: random_square` with `out_size: N`) and the same burst_size /
window_size / vjepa_window_size / wan_num_frames.

patch_aspect_ratios: true -- lifts the uniform-shape requirement across
sources (never within one batch, which torch.stack still needs). Every batch
comes from ONE source, and the crop SHAPE varies across batches instead of
being fixed -- so e.g. mixing square Adobe240/GoPro/FMO crops with OMoBlur's
much wider (5.33:1) ones just works, no changes needed to any of those
dataset classes. Uses the SAME disk-IO-amortization producer pattern as
FullResPatchifierDataset (see that file's docstring for the original of this
pattern): load `batch_size` samples from ONE source ONCE (the expensive disk
read + decode -- "keep the clip in memory"), then draw `patches_per_group`
DIFFERENT (crop_h, crop_w) shapes from that SAME in-memory group before
loading the next group. Which source a group comes from is still governed by
the normal `weight:` proportions. This mode makes MixedDataset an
IterableDataset (see MixedDataset.__new__) -- models/base_model.py's
setup_dataloaders already handles that (batch_size=None, no shuffle, same as
FullResPatchifierDataset), no separate wrapper needed; MixedDataset already
IS the wrapper, this is just another mode of it, not a second class stacked
on top.

    patch_aspect_ratios: true
    batch_size: 4                # DataLoader is built with batch_size=None in this mode --
                                  # this is the REAL per-step batch size instead
    patches_per_group: 10        # crops drawn per one group load (the disk-IO amortization knob)
    crop_size: 512                # target crop area ~= crop_size^2
    crop_aspect_min: 0.5          # crop_w / crop_h
    crop_aspect_max: 2.0
    shape_divisor: 32             # Wan's vae_scale_factor_spatial(16) * patch_size(2)

    # Optional, TRAIN-only, all default off (patch_aspect_ratios mode only; every draw is per BATCH):
    crop_size_choices: [384, 512, 640, 704]   # per-batch target crop side (area = side^2), overrides crop_size
    frame_counts: [5, 9]          # per-batch temporal CENTER-crop of `frame_keys` (each must be 4n+1; <= source F)
    frame_keys: [wan_frames]
    augment:
      hflip: 0.5                  # p, applied to EVERY array key of the batch (range-agnostic)
      vflip: 0.0
      time_reverse: 0.5           # p, flips the T axis of `frame_keys` only (the blur is time-symmetric)
      scale_range: [0.75, 1.5]    # per-batch source-crop/target ratio, then resize to the target shape
"""
import random as _random
from typing import List

from torch.utils import data as data
from torch.utils.data import default_collate

import numpy as np
import torch


def _build_sources(opt):
    """Shared by both modes: instantiate every `sources` entry, normalize weights, and build the
    per-source label list (see MixedDataset.__init__'s own comment on duplicate-class labels)."""
    from dataset import create_dataset  # local import: avoids a circular import at module load

    sources = opt.get("sources", None)
    if not sources:
        raise ValueError("MixedDataset requires a non-empty `sources` list")

    datasets, weights, names = [], [], []
    for src in sources:
        # `src` is whatever the config wrapper produced (DictAsMember here, possibly DictConfig
        # elsewhere) -- access via .get() only, never dict(...) / `in` / [...], which DictAsMember
        # silently answers against an empty dict (see utils/misc.py and
        # findings/Wan22VJEPADynamics_findings.md's 2026-09-21 entry).
        w = float(src.get("weight", 1.0))
        if w < 0:
            raise ValueError(f"source weight must be >= 0, got {w}")
        ds = create_dataset(src)
        datasets.append(ds)
        weights.append(w)
        label = str(src.get("name", src.get("type", "?")))
        if label in names:
            label = f"{label}_{sum(1 for n in names if n.split('#')[0] == label) + 1}"
        names.append(label)

    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("MixedDataset source weights sum to 0")
    weights = [w / total_w for w in weights]
    return datasets, weights, names


class MixedDataset(data.Dataset):
    def __new__(cls, opt):
        # Dispatch to the IterableDataset sibling for patch_aspect_ratios mode (see the module
        # docstring) -- default False takes this same path as before, byte-for-byte unaffected.
        if bool(opt.get("patch_aspect_ratios", False)):
            return _MixedDatasetPatchified(opt)
        return super().__new__(cls)

    def __init__(self, opt):
        super().__init__()
        self.datasets, weights, names = _build_sources(opt)

        epoch_size = opt.get("epoch_size", None)
        epoch_size = int(epoch_size) if epoch_size is not None else sum(len(d) for d in self.datasets)

        # The plan fixes the source MIX exactly; which sample each slot draws
        # depends on `use_shuffle`.
        #
        # The original version used `(i, j % len(ds)) for j in range(n_i)`,
        # which is a PREFIX, not a sample: when n_i < len(ds) only sub-indices
        # 0..n_i-1 were ever reachable, and because the plan is built once in
        # __init__ and reused for the whole run (not rebuilt per epoch), the
        # rest of the pool was unreachable FOREVER -- 52.5% of Adobe240 at the
        # shipped 4000x0.50 vs a 4214 pool, and because each source interleaves
        # scenes round-robin, the unreachable half was the late part of every
        # scene. Validation was worse: n_i=2 meant literally burst 0 of the
        # first two scenes, every time.
        #
        #   use_shuffle: true  (training)   -> the slot's sample is drawn at
        #       access time, so the full pool is reachable across the run and
        #       epochs differ. The MIX RATIO stays exact because the number of
        #       slots per source is still fixed here.
        #   use_shuffle: false (validation) -> deterministic indices spread
        #       EVENLY across the whole pool, so a small val set spans many
        #       scenes instead of the first few, and is identical at every
        #       checkpoint (which is what makes val curves comparable).
        self.resample_per_access = bool(opt.get("use_shuffle", True))
        self.plan: List[tuple] = []
        for i, (ds, w) in enumerate(zip(self.datasets, weights)):
            n_i = int(round(epoch_size * w))
            if n_i > 0 and len(ds) == 0:
                raise ValueError(f"source {i} ({names[i]}) has 0 samples but weight {w}")
            if self.resample_per_access:
                self.plan.extend((i, None) for _ in range(n_i))
            elif n_i <= len(ds):
                self.plan.extend((i, (j * len(ds)) // n_i) for j in range(n_i))
            else:
                self.plan.extend((i, j % len(ds)) for j in range(n_i))

        rng = _random.Random(int(opt.get("seed", 42)))
        rng.shuffle(self.plan)

        self.source_names = names
        counts = [sum(1 for s, _ in self.plan if s == i) for i in range(len(self.datasets))]
        print(
            "Initialized MixedDataset with "
            + ", ".join(
                f"{names[i]}: {counts[i]} samples/epoch ({counts[i] / max(len(self.plan), 1):.0%}, "
                f"pool={len(self.datasets[i])}, {counts[i] / max(len(self.datasets[i]), 1):.1f}x oversample)"
                for i in range(len(self.datasets))
            )
            + f" -- epoch_size={len(self.plan)}"
        )

    def __getattr__(self, name):
        """Delegate unknown attributes to the first source dataset.

        Models reach into the dataset for its key names -- base_model.py's
        validation passes `dataset.lq_key`/`gt_key` into validate_step, and
        BlurJepaDynamics_model's burst path reads `ds.burst_key` /
        `ds.burst_frames_key`. A wrapper dataset that does not forward those
        raises AttributeError deep inside the training loop, typically at the
        FIRST VALIDATION rather than at startup, so it can burn a queue slot
        and model-load time before failing.

        Safe because __getattr__ only fires for attributes not found normally,
        so it never shadows this class's own state. All sources are required to
        share an identical output contract anyway (see the module docstring),
        so source 0's key names are the contract.
        """
        if name.startswith("_") or "datasets" not in self.__dict__:
            raise AttributeError(name)
        try:
            return getattr(self.__dict__["datasets"][0], name)
        except (AttributeError, IndexError):
            raise AttributeError(
                f"{type(self).__name__!r} and its source datasets have no attribute {name!r}"
            ) from None

    def __len__(self):
        return len(self.plan)

    def collate_fn(self, batch):
        return default_collate(batch)

    def __getitem__(self, idx):
        src_i, sample_i = self.plan[idx]
        ds = self.datasets[src_i]
        if sample_i is None:  # training: draw from the whole pool
            sample_i = _random.randrange(len(ds))
        sample = ds[sample_i]
        # Tag the originating source so downstream metric logging can report
        # per-source numbers. Without this a mixed val set collapses into one
        # untagged mean, which defeats the point of probing two different
        # capabilities (retention vs new skill) in the same validation pass.
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["source"] = self.source_names[src_i]
        return sample


class _MixedDatasetPatchified(data.IterableDataset):
    """The patch_aspect_ratios: true mode of MixedDataset -- see MixedDataset.__new__ and the
    module docstring. Not meant to be referenced directly in a config (type: MixedDataset always
    dispatches here when the flag is set)."""

    def __init__(self, opt):
        super().__init__()
        self.datasets, weights, self.source_names = _build_sources(opt)
        self.weights = weights
        self.is_train = bool(opt.get("is_train", bool(opt.get("use_shuffle", True))))

        self.batch_size = int(opt.get("batch_size", 4))
        self.patches_per_group = int(opt.get("patches_per_group", 10))
        self.crop_size = int(opt.get("crop_size", 512))
        self.crop_area = self.crop_size * self.crop_size
        self.crop_aspect_min = float(opt.get("crop_aspect_min", 0.5))
        self.crop_aspect_max = float(opt.get("crop_aspect_max", 2.0))
        self.shape_divisor = int(opt.get("shape_divisor", 32))
        self.seed = int(opt.get("seed", 1234))
        # opt values may be DictAsMember/ListConfig: read via .get() and coerce with list()/float() only on leaves
        _choices = opt.get("crop_size_choices", None)
        self.crop_size_choices = [int(c) for c in _choices] if _choices else None
        _fc = opt.get("frame_counts", None)
        self.frame_counts = [int(c) for c in _fc] if _fc else None
        _fk = opt.get("frame_keys", None)
        self.frame_keys = [str(k) for k in _fk] if _fk else ["wan_frames"]
        _aug = opt.get("augment", None)
        _g = (lambda k, d: _aug.get(k, d)) if _aug is not None else (lambda k, d: d)
        self.aug_hflip = float(_g("hflip", 0.0))
        self.aug_vflip = float(_g("vflip", 0.0))
        self.aug_time_reverse = float(_g("time_reverse", 0.0))
        _sr = _g("scale_range", None)
        self.aug_scale_range = [float(_sr[0]), float(_sr[1])] if _sr else None
        # Re-synthesise the blur as the mean of exactly the F frames the clip keeps (exposure == clip span). OFF by default: the
        # eval blur is the sources' own 11-frame sRGB-space mean, so enabling this changes the blur formation vs eval. Use it only
        # with frame_counts < the source window, together with reblur_target=gt_mean. blur_space: srgb | linear.
        self.blur_from_clip = bool(opt.get("blur_from_clip", False))
        self.blur_space = str(opt.get("blur_space", "srgb"))
        self.blur_key = str(opt.get("blur_key", "blur"))
        self._aug_on = any([self.aug_hflip, self.aug_vflip, self.aug_time_reverse, self.aug_scale_range])
        if not self.is_train and (self.crop_size_choices or self.frame_counts or self._aug_on):
            raise ValueError("crop_size_choices / frame_counts / augment are train-only (this MixedDataset is not is_train)")

        epoch_size = opt.get("epoch_size", None)
        self.epoch_size = int(epoch_size) if epoch_size is not None else sum(len(d) for d in self.datasets)
        print(
            f"Initialized MixedDataset (patch_aspect_ratios) with "
            + ", ".join(f"{n}: weight={w:.2f}, pool={len(d)}" for n, w, d in zip(self.source_names, weights, self.datasets))
            + f" -- epoch_size={self.epoch_size}, batch_size={self.batch_size}, "
              f"patches_per_group={self.patches_per_group}, crop_area~{self.crop_size}^2, "
              f"crop_aspect=[{self.crop_aspect_min},{self.crop_aspect_max}]"
        )

    def __len__(self):
        n_groups = max(1, self.epoch_size // self.batch_size)
        return n_groups * self.patches_per_group

    def _round_to_divisor(self, x, max_x):
        x = (min(x, max_x) // self.shape_divisor) * self.shape_divisor
        return max(self.shape_divisor, x)

    def _sample_crop_shape(self, rng, max_h, max_w, crop_area=None):
        crop_area = self.crop_area if crop_area is None else crop_area
        aspect = rng.uniform(self.crop_aspect_min, self.crop_aspect_max)
        ch = int(round((crop_area * aspect) ** 0.5))
        cw = int(round((crop_area / aspect) ** 0.5))
        return self._round_to_divisor(ch, max_h), self._round_to_divisor(cw, max_w)

    @staticmethod
    def _resize_hw(t, size):
        """Resize the last two axes of a tensor of any rank to `size`=(h, w)."""
        if tuple(t.shape[-2:]) == tuple(size):
            return t
        lead = t.shape[:-2]
        x = t.reshape(-1, 1, *t.shape[-2:])
        if x.is_floating_point():
            x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
        else:
            x = torch.nn.functional.interpolate(x, size=size, mode="nearest")
        return x.reshape(*lead, *size)

    def _temporal_center_crop(self, batch, rng):
        """Truncate `frame_keys` ([B,T,...]) to a random allowed frame count, keeping the clip CENTERED on the
        blur window (the source emits wan_start=(window-F)/2, so a centered sub-crop is still centered)."""
        f = rng.choice(self.frame_counts)
        for k in self.frame_keys:
            if k not in batch:
                continue
            t = batch[k].shape[1]
            if f > t:
                raise ValueError(f"frame_counts has {f} > source {k} length {t}")
            start = (t - f) // 2
            batch[k] = batch[k][:, start:start + f]

    def _resynth_blur(self, batch):
        fk = self.frame_keys[0]
        v = (batch[fk].float() + 1.0) / 2.0                      # [B,F,3,H,W] in [0,1] (pm1 source)
        if self.blur_space == "linear":
            v = v.clamp(0, 1).pow(2.2)
        m = v.mean(dim=1)
        if self.blur_space == "linear":
            m = m.clamp(min=0).pow(1 / 2.2)
        batch[self.blur_key] = (m * 2.0 - 1.0).to(batch[self.blur_key].dtype) if self.blur_key in batch else m * 2.0 - 1.0

    def _augment_batch(self, batch, rng):
        has_flow = any("flow" in k for k in batch)
        if self.aug_hflip and not has_flow and rng.random() < self.aug_hflip:
            for k in batch:
                batch[k] = batch[k].flip(-1)
        if self.aug_vflip and not has_flow and rng.random() < self.aug_vflip:
            for k in batch:
                batch[k] = batch[k].flip(-2)
        if self.aug_time_reverse and rng.random() < self.aug_time_reverse:
            for k in self.frame_keys:
                if k in batch:
                    batch[k] = batch[k].flip(1)

    @staticmethod
    def _array_keys(sample: dict):
        """Every key whose value is array-like with >= 2 dims, i.e. it has (H, W) as its last two
        axes per this codebase's channel-first convention -- everything else (name, source,
        scalars) is metadata and passes through uncropped/unstacked."""
        return [k for k, v in sample.items() if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 2]

    def __iter__(self):
        # Validation must draw the exact same sequence of source picks/groups/crops every time
        # __iter__ runs (each call to validation() re-iterates from scratch) -- same reasoning as
        # FullResPatchifierDataset's __iter__.
        worker_info = data.get_worker_info()
        if self.is_train:
            rng = _random.Random(worker_info.seed) if worker_info is not None else _random
        else:
            rng = _random.Random(self.seed)

        n_groups = max(1, self.epoch_size // self.batch_size)
        if worker_info is not None:
            # Disjoint group shard per worker -- IterableDataset has no built-in batching/sharding.
            group_range = range(worker_info.id, n_groups, worker_info.num_workers)
        else:
            group_range = range(n_groups)

        for _ in group_range:
            src_i = rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
            ds = self.datasets[src_i]
            group_idx = [rng.randrange(len(ds)) for _ in range(self.batch_size)]
            loaded = [ds[i] for i in group_idx]  # the expensive disk read + decode, ONCE per group

            array_keys = self._array_keys(loaded[0])
            ref_key = array_keys[0]
            native_hw = [(np.asarray(s[ref_key]).shape[-2], np.asarray(s[ref_key]).shape[-1]) for s in loaded]
            min_h = min(h for h, _ in native_hw)
            min_w = min(w for _, w in native_hw)

            for _ in range(self.patches_per_group):
                area = (rng.choice(self.crop_size_choices) ** 2) if self.crop_size_choices else None
                ch, cw = self._sample_crop_shape(rng, min_h, min_w, area)
                # scale augmentation: crop a (ch*s, cw*s) region of the source and resize it to (ch, cw)
                # ONE isotropic zoom factor (never per-axis clamping, which stretched the image): s_eff in [s_lo', s_hi'] is
                # limited so the source crop (ch*s_eff, cw*s_eff) still fits the source. s<1 (zoom-IN, crop smaller than the
                # target then upsampled) is allowed down to scale_range[0]; the draw is in log space so zoom-in/out are symmetric.
                if self.aug_scale_range:
                    lo, hi = self.aug_scale_range
                    s = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                    s = min(s, min_h / ch, min_w / cw)
                else:
                    s = 1.0
                sh, sw = max(self.shape_divisor // 2, int(round(ch * s))), max(self.shape_divisor // 2, int(round(cw * s)))
                sh, sw = min(sh, min_h), min(sw, min_w)
                collected = {k: [] for k in array_keys}
                for sample, (h, w) in zip(loaded, native_hw):
                    top = rng.randint(0, h - sh) if h > sh else 0
                    left = rng.randint(0, w - sw) if w > sw else 0
                    for k in array_keys:
                        cropped = sample[k][..., top:top + sh, left:left + sw]
                        if isinstance(cropped, np.ndarray):
                            cropped = torch.from_numpy(np.ascontiguousarray(cropped))
                        if (sh, sw) != (ch, cw):
                            cropped = self._resize_hw(cropped, (ch, cw))
                        collected[k].append(cropped)
                # meta_keys (name, source, ...) are DROPPED here, not yielded -- accelerate's
                # DataLoaderDispatcher (used by accelerator.prepare(self.dataloader), which wraps
                # even a single-process run) prefetches a step ahead and concatenates same-key
                # values across the prefetched batches; given a plain Python list of strings it
                # recurses into it expecting tensors and crashes ("Can only concatenate tensors but
                # got <class 'str'>") -- confirmed by an actual smoke run, not a guess. Only
                # torch.Tensor leaves survive that dispatch, so this mode can only carry array
                # values. This is fine for TRAINING (sample["name"]/["source"] are read only inside
                # validate_step, never optimize_parameters -- see the grep in the PR description);
                # it means patch_aspect_ratios mode can't (yet) be used for VALIDATION without losing
                # per-clip stem naming and per-source metric tagging -- a known limitation, not
                # silently broken.
                batch = {}
                for k in array_keys:
                    try:
                        batch[k] = torch.stack(collected[k])
                    except RuntimeError as e:
                        # Only the last two (spatial) axes are guaranteed uniform by this crop --
                        # a LEADING-axis mismatch (e.g. OMoBlurDataset's `frames` window length
                        # varying with blur_lengths) means this source's own samples don't already
                        # share a non-spatial shape, which this cannot fix (it only knows H,W).
                        raise RuntimeError(
                            f"MixedDataset (patch_aspect_ratios): source {self.source_names[src_i]!r} "
                            f"key {k!r} has a non-spatial shape mismatch within one group "
                            f"({[tuple(t.shape) for t in collected[k]]}) -- fix that source's config "
                            f"so every sample emits the same shape outside its last two (H,W) axes."
                        ) from e
                if self.frame_counts:
                    self._temporal_center_crop(batch, rng)
                if self.blur_from_clip:
                    self._resynth_blur(batch)
                if self._aug_on:
                    self._augment_batch(batch, rng)
                yield batch

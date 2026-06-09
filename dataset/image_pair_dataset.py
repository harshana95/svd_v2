import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils import data as data


class ImagePairDataset(data.Dataset):
    """Generic loader for paired image-to-image datasets.

    Expected layout (supports nested scene folders):
      <data_dir>/<split>/<scene>/<lq_folder>/* and <data_dir>/<split>/<scene>/<gt_folder>/*
    or:
      <data_dir>/<scene>/<lq_folder>/* and <data_dir>/<scene>/<gt_folder>/*

    If `recursive` is True, any folder under the split root containing both
    `<lq_folder>` and `<gt_folder>` is treated as a valid pair root.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.split = opt.get("split", "train")

        self.gt_key = opt.get("gt_key", "gt")
        self.lq_key = opt.get("lq_key", "lq")
        self.return_name = opt.get("return_name", True)

        self.lq_folder = opt.get("lq_folder", "lq")
        self.gt_folder = opt.get("gt_folder", "gt")
        self.recursive = opt.get("recursive", True)
        self.match_by_stem = opt.get("match_by_stem", False)

        self.crop_h = opt.get("crop_h", None)
        self.crop_w = opt.get("crop_w", None)
        self.random_crop = opt.get("random_crop", True)

        self.extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in opt.get("extensions", [".png", ".jpg", ".jpeg", ".bmp", ".webp"])
        )

        self.samples = self._collect_pairs()
        self.samples = self._apply_sample_range(self.samples)
        if len(self.samples) == 0:
            raise ValueError(
                f"No valid image pairs found in '{self.data_dir}' with split '{self.split}'. "
                f"Expected '{self.lq_folder}' and '{self.gt_folder}' subdirectories with matching files."
            )

        print(f"Initialized ImagePairDataset ({self.split}) with {len(self.samples)} image pairs")

    def _split_root(self) -> str:
        split_root = os.path.join(self.data_dir, self.split)
        return split_root if os.path.isdir(split_root) else self.data_dir

    def _is_image_file(self, path: str) -> bool:
        return os.path.isfile(path) and path.lower().endswith(self.extensions)

    def _collect_pair_roots(self, split_root: str) -> List[str]:
        pair_roots: List[str] = []

        if self.recursive:
            for root, dirs, _ in os.walk(split_root):
                if self.lq_folder in dirs and self.gt_folder in dirs:
                    pair_roots.append(root)
                # Do not recurse into the image folders themselves.
                dirs[:] = [d for d in dirs if d not in {self.lq_folder, self.gt_folder}]
        else:
            if os.path.isdir(os.path.join(split_root, self.lq_folder)) and os.path.isdir(
                os.path.join(split_root, self.gt_folder)
            ):
                pair_roots.append(split_root)
            else:
                for scene in sorted(os.listdir(split_root)):
                    scene_dir = os.path.join(split_root, scene)
                    if not os.path.isdir(scene_dir):
                        continue
                    if os.path.isdir(os.path.join(scene_dir, self.lq_folder)) and os.path.isdir(
                        os.path.join(scene_dir, self.gt_folder)
                    ):
                        pair_roots.append(scene_dir)

        return sorted(pair_roots)

    def _collect_pairs_for_root(self, root: str) -> List[Tuple[str, str]]:
        lq_dir = os.path.join(root, self.lq_folder)
        gt_dir = os.path.join(root, self.gt_folder)
        pairs: List[Tuple[str, str]] = []

        if not self.match_by_stem:
            for file_name in sorted(os.listdir(lq_dir)):
                lq_path = os.path.join(lq_dir, file_name)
                gt_path = os.path.join(gt_dir, file_name)
                if not self._is_image_file(lq_path) or not self._is_image_file(gt_path):
                    continue
                pairs.append((lq_path, gt_path))
            return pairs

        gt_stem_map: Dict[str, str] = {}
        for file_name in sorted(os.listdir(gt_dir)):
            gt_path = os.path.join(gt_dir, file_name)
            if not self._is_image_file(gt_path):
                continue
            gt_stem_map[os.path.splitext(file_name)[0]] = gt_path

        for file_name in sorted(os.listdir(lq_dir)):
            lq_path = os.path.join(lq_dir, file_name)
            if not self._is_image_file(lq_path):
                continue
            stem = os.path.splitext(file_name)[0]
            gt_path: Optional[str] = gt_stem_map.get(stem)
            if gt_path is None:
                continue
            pairs.append((lq_path, gt_path))
        return pairs

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        split_root = self._split_root()
        pairs: List[Tuple[str, str]] = []
        for root in self._collect_pair_roots(split_root):
            pairs.extend(self._collect_pairs_for_root(root))
        return pairs

    def _apply_sample_range(self, samples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        split_range = self.opt.get("split_range", None)
        if split_range is None:
            split = self.opt.get("split_ratio", None)
            if split is None:
                return samples
            start, end = split
            n = len(samples)
            return samples[int(start * n) : int(end * n)]

        n = len(samples)
        start = int(float(split_range[0]) / 100.0 * n)
        end = int(float(split_range[1]) / 100.0 * n)
        return samples[start:end]

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        image = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        image = image.transpose(2, 0, 1) / 127.5 - 1.0
        return image

    def _crop_pair(self, gt: np.ndarray, lq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.crop_h is None or self.crop_w is None:
            return gt, lq

        _, h, w = gt.shape
        if h < self.crop_h or w < self.crop_w:
            return gt, lq

        if self.random_crop:
            top = int(np.random.randint(0, h - self.crop_h + 1))
            left = int(np.random.randint(0, w - self.crop_w + 1))
        else:
            top = (h - self.crop_h) // 2
            left = (w - self.crop_w) // 2

        gt = gt[:, top : top + self.crop_h, left : left + self.crop_w]
        lq = lq[:, top : top + self.crop_h, left : left + self.crop_w]
        return gt, lq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lq_path, gt_path = self.samples[int(idx)]
        gt = self._load_image(gt_path)
        lq = self._load_image(lq_path)
        gt, lq = self._crop_pair(gt, lq)

        sample = {
            self.lq_key: lq,
            self.gt_key: gt,
        }

        if self.return_name:
            sample["name"] = os.path.relpath(lq_path, self._split_root())
        return sample

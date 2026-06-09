import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils import data as data


class GoProDataset(data.Dataset):
    """Dataset loader for GoPro deblurring pairs.

    Expected layouts:
    1) <data_dir>/<split>/<scene>/blur/*.png and <data_dir>/<split>/<scene>/sharp/*.png
    2) <data_dir>/<scene>/blur/*.png and <data_dir>/<scene>/sharp/*.png
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.split = opt.get("split", "train")
        self.gt_key = opt.get("gt_key", "gt")
        self.lq_key = opt.get("lq_key", "blur")
        self.return_name = opt.get("return_name", True)
        self.crop_h = opt.get("crop_h", None)
        self.crop_w = opt.get("crop_w", None)
        self.random_crop = opt.get("random_crop", True)

        self.samples = self._collect_pairs()
        self.samples = self._apply_sample_range(self.samples)
        if len(self.samples) == 0:
            raise ValueError(
                f"No valid GoPro pairs found in '{self.data_dir}' with split '{self.split}'. "
                "Expected folders containing 'blur' and 'sharp' subdirectories."
            )
        print(f"Initialized GoProDataset ({self.split}) with {len(self.samples)} image pairs")

    def _split_root(self) -> str:
        split_root = os.path.join(self.data_dir, self.split)
        return split_root if os.path.isdir(split_root) else self.data_dir

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        split_root = self._split_root()
        pairs: List[Tuple[str, str]] = []

        for scene in sorted(os.listdir(split_root)):
            scene_dir = os.path.join(split_root, scene)
            if not os.path.isdir(scene_dir):
                continue

            blur_dir = os.path.join(scene_dir, "blur")
            sharp_dir = os.path.join(scene_dir, "sharp")
            if not os.path.isdir(blur_dir) or not os.path.isdir(sharp_dir):
                continue

            for file_name in sorted(os.listdir(blur_dir)):
                blur_path = os.path.join(blur_dir, file_name)
                sharp_path = os.path.join(sharp_dir, file_name)
                if not os.path.isfile(blur_path) or not os.path.isfile(sharp_path):
                    continue
                pairs.append((blur_path, sharp_path))
        return pairs

    def _apply_sample_range(self, samples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        split_range = self.opt.get("split_range", None)
        if split_range is None:
            split = self.opt.get("split_ratio", None)
            if split is None:
                return samples
            start, end = split
            n = len(samples)
            return samples[int(start * n): int(end * n)]

        n = len(samples)
        start = int(float(split_range[0]) / 100.0 * n)
        end = int(float(split_range[1]) / 100.0 * n)
        return samples[start:end]

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        image = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        image = image.transpose(2, 0, 1) / 127.5 - 1.0
        return image

    def _crop_pair(self, gt: np.ndarray, lq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_h is None or self.crop_w is None:
            return gt, lq
        _, h, w = gt.shape
        if h <= self.crop_h and w <= self.crop_w:
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
        blur_path, sharp_path = self.samples[int(idx)]
        gt = self._load_image(sharp_path)
        lq = self._load_image(blur_path)
        gt, lq = self._crop_pair(gt, lq)
        sample = {
            self.lq_key: lq,
            self.gt_key: gt,
        }
        if self.return_name:
            sample["name"] = os.path.relpath(blur_path, self._split_root())
        return sample

import copy
import os

from PIL import Image
from torch.utils import data as data

from utils import get_dataset_util
from utils.dataset_utils import DictWrapper
from utils.misc import DictAsMember


def _transform_opt_to_dict(transform_opt):
    if transform_opt is None:
        return {}
    if hasattr(transform_opt, "__dict__"):
        return {k: transform_opt[k] for k in transform_opt.__dict__ if not k.startswith("_")}
    return dict(transform_opt)


class FixedImageDataset(data.Dataset):
    """Load a single image from disk and apply the standard YAML transform pipeline."""

    def __init__(self, opt):
        self.opt = opt
        self.gt_key = opt.gt_key
        self.lq_key = opt.lq_key

        image_path = opt.image_path
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"FixedImageDataset image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        sample = {self.gt_key: image, self.lq_key: image}
        self.sample = self._apply_transforms(sample, opt)
        print(f"Initialized FixedImageDataset from {image_path}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample

    def _build_transforms(self, opt):
        transforms_list = []
        gt_key, lq_key = self.gt_key, self.lq_key
        if opt.get("transforms") is None:
            return transforms_list

        for transform_name in opt.get("transforms"):
            transform_opt = opt.transforms[transform_name]
            if transform_opt is None:
                transform_opt = {}
            else:
                transform_opt = copy.deepcopy(_transform_opt_to_dict(transform_opt))

            keys = transform_opt.pop("keys", None)
            keys_new = transform_opt.pop("keys_new", None)
            if keys is None:
                keys = opt.common_transforms_keys
            if keys_new is None:
                keys_new = []

            keys = [key.replace("gt_key", gt_key) for key in keys]
            keys = [key.replace("lq_key", lq_key) for key in keys]
            keys_new = [key.replace("gt_key", gt_key) for key in keys_new]
            keys_new = [key.replace("lq_key", lq_key) for key in keys_new]

            transform = get_dataset_util(
                transform_name.split("--")[0],
                DictAsMember(**transform_opt) if transform_opt else {},
            )
            transforms_list.append(DictWrapper(transform, keys, keys_new))

        print("All transformations for the dataset ")
        for tr in transforms_list:
            print(tr.f, tr.keys, tr.new_keys)
        return transforms_list

    def _apply_transforms(self, sample, opt):
        for transform in self._build_transforms(opt):
            batch = {key: [value] for key, value in sample.items()}
            batch = transform(batch)
            sample = {key: values[0] for key, values in batch.items()}
        return sample

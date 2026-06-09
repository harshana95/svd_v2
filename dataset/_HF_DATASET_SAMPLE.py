import cv2
import numpy as np
import h5py
import datasets
from datasets import BuilderConfig, Features, Value, SplitGenerator, Array2D, Image, Sequence
import hashlib
import os
import json
from os.path import join

from tqdm import tqdm

class CustomHFDataset(datasets.GeneratorBasedBuilder):
    fpath = os.path.realpath(__file__)
    js = json.loads(open(str(fpath)[:-3]+'.json').read())
    data_dir = os.path.dirname(js['original file path'])
    
    VERSION = datasets.Version("1.2.1")

    BUILDER_CONFIGS = [
        BuilderConfig(name="default", version=VERSION, description="Default configuration for NYUv2 dataset"),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features({
            "gt": Image(decode=True),
            "blur": Image(decode=True),
        })

        return datasets.DatasetInfo(
            description="custom HF dataset",
            features=features,
        )

    def _split_generators(self, dl_manager):
        train_pairs = self._collect_file_pairs("train")
        val_pairs = self._collect_file_pairs("val")
        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": datasets.Split.TRAIN,
                    "file_pairs": train_pairs,
                },
            ),
            SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": datasets.Split.VALIDATION,
                    "file_pairs": val_pairs,
                },
            ),
        ]

    def _collect_file_pairs(self, split_dir):
        gt_dir = join(self.data_dir, split_dir, "gt")
        blur_dir = join(self.data_dir, split_dir, "blur")
        files = sorted(os.listdir(gt_dir))
        file_pairs = []
        for fname in files:
            gt_path = join(gt_dir, fname)
            blur_path = join(blur_dir, fname)
            if os.path.isfile(gt_path) and os.path.isfile(blur_path):
                file_pairs.append((gt_path, blur_path))
        return file_pairs
    
    def _generate_examples(self, split, file_pairs):
        # write your own logic to generate the dataset
        print(f"found {len(file_pairs)} files for split {split}")

        def get_dict(p1, p2):
            return {
                    "gt": cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB),
                    "blur": cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB),
                }

        split_prefix = "train" if split == datasets.Split.TRAIN else "val"
        for idx, (p1, p2) in enumerate(tqdm(file_pairs)):
            sample_key = f"{split_prefix}_{idx}_{os.path.basename(p1)}"
            yield sample_key, get_dict(p1, p2)
        
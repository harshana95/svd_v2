"""NYU Depth Dataset V2"""

import cv2
import numpy as np
import h5py
import datasets
from datasets import BuilderConfig, Features, Value, SplitGenerator, Array2D, Image, Sequence
import hashlib
import os
import json
from os.path import join

class NYUv2(datasets.GeneratorBasedBuilder):
    """NYU Depth Dataset V2"""
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
            "image": Image(decode=True),
            "depth": Image(decode=True),
            "annotation": Value("string"),
        })

        return datasets.DatasetInfo(
            description="Image depth and annotation dataset",
            features=features,
        )

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": datasets.Split.TRAIN,
                },
            ),
            SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": datasets.Split.VALIDATION,
                },
            ),
        ]
    
    @staticmethod
    def get_first_file(path):
        return os.listdir(path)[0]
    
    def _generate_examples(self, split):
        train_percent = 0.99
        
        folders = [entry for entry in os.listdir(self.data_dir) if os.path.isdir(join(self.data_dir, entry))]
        keys = [entry for entry in os.listdir(join(self.data_dir, folders[0])) if os.path.isdir(join(self.data_dir, folders[0], entry))]
        train_folders = folders[:int(len(folders)*train_percent)]
        val_folders = folders[int(len(folders)*train_percent):]

        def get_dict(p):
            return {
                    "image": cv2.cvtColor(cv2.imread(join(p, "image", self.get_first_file(join(p, "image")))), cv2.COLOR_BGR2RGB),
                    "depth": cv2.cvtColor(cv2.imread(join(p, "depth_bfx", self.get_first_file(join(p, "depth_bfx")))), cv2.COLOR_BGR2RGB),
                    "annotation": open(join(p, "annotation", "index.json")).read(),
                }

        if split == datasets.Split.TRAIN:
            for idx in range(len(train_folders)):
                p = join(self.data_dir, train_folders[idx])
                yield idx, get_dict(p)
        else:
            for idx in range(len(val_folders)):
                p = join(self.data_dir, val_folders[idx])
                yield idx, get_dict(p)
        
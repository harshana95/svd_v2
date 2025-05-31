from torch.utils.data import Dataset
import numpy as np

class ZipDatasetWrapper(Dataset):
    def __init__(self, dataset_dict, transforms, random=False, **kwargs):
        super().__init__(**kwargs)
        self.datasets = dataset_dict
        self.transforms = transforms
        self.random = random
        self.lengths = {}
        tmp = -1
        # check for duplicate keys in the given datasets
        ret = set()
        for dataset_key in self.datasets.keys():
            length = len(self.datasets[dataset_key])
            self.lengths[dataset_key] = length
            if not self.random:
                if tmp >= 0 and tmp != length:
                    print(Warning(f"Lengths of the datasets are not equal. last {tmp} current {length} ({dataset_key})"))
            tmp = length

            sample = self.datasets[dataset_key][0]
            for k, v in sample.items():
                k = f"{k}_{dataset_key}"
                if k in ret and k != 'idx':
                    raise Exception(f'Duplicate key: {k}')
                ret.add(k)

    def __getitem__(self, index):
        ret = {}
        indices = {}
        for dataset_key in self.datasets.keys():
            idx = index%self.lengths[dataset_key] if not self.random else np.random.randint(self.lengths[dataset_key])
            indices[dataset_key] = idx
            sample = self.datasets[dataset_key][idx]
            for k, v in sample.items():
                k = f"{k}_{dataset_key}"
                ret[k] = v
        ret = self.transforms(ret)
        return ret

    def __len__(self):
        return max(self.lengths.values())

import json
import os
import datasets

class ImagePairDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    fpath = os.path.realpath(__file__)
    js = json.loads(open(str(fpath)[:-3]+'.json').read())
    data_dir = os.path.dirname(js['original file path'])
    # print(data_dir, os.getcwd(), os.path.realpath(__file__))
    folders = os.listdir(os.path.join(data_dir, "train"))
    print(folders)

    def _info(self):

        return datasets.DatasetInfo(
            description="An image dataset",
            features=datasets.Features({f: datasets.Image() for f in self.folders}),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={f"{f}": os.path.join(self.data_dir, "train", f) for f in self.folders},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={f"{f}": os.path.join(self.data_dir, "val", f) for f in self.folders},
            ),
        ]


    def _generate_examples(self, **kwargs):
        dirs = list(kwargs.keys())
        files = {f: sorted(os.listdir(kwargs[f])) for f in dirs}

        for idx in range(len(files[dirs[0]])):
            
            paths = {f: os.path.join(kwargs[f], files[f][idx]) for f in dirs}
            yield idx, paths
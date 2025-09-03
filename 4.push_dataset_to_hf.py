import os
import shutil
from os.path import join

from datasets import disable_caching, load_dataset
from huggingface_hub import HfApi
ds_dir = "/depot/chan129/data/Deconvolution/fg_bg_data/ForegroundDataset"
md_dir = join(ds_dir, 'metadata')
os.makedirs(md_dir, exist_ok=True)
dataset_name = os.path.basename(ds_dir)

disable_caching()

shutil.copyfile('./dataset/loading_script.py', join(ds_dir, f'{dataset_name}.py'))
shutil.rmtree('./.cachehf', ignore_errors=True)
dataset = load_dataset(ds_dir, trust_remote_code=True, cache_dir='./.cachehf')
shutil.rmtree('./.cachehf', ignore_errors=True)
print(f"Length of the created dataset {len(dataset['train'])}")

repoid = f"harshana95/{dataset_name}"
dataset.push_to_hub(repoid, num_shards={'train': 10, 'val': 1})

for batch in dataset['train']:
    print(batch)
    break
    


api = HfApi()
api.upload_folder(
    folder_path=md_dir,
    repo_id=repoid,
    path_in_repo="metadata",
    repo_type="dataset",

)


"""One-time warmup: build the Deblur_MIOR HF `datasets` arrow cache (raw
generation + the transform-pipeline `.map()`, which materializes over
`repeat`-expanded rows) so a later multi-rank job finds it already cached
instead of every rank rebuilding it -- or, worse, all ranks fighting over
building it at once.

Must run on a compute node (not the login node -- `datasets.map(num_proc=...)`
forks worker subprocesses that hit the login node's RLIMIT_NPROC and hang).
See scripts/alcf/sjob_warm_dataset_cache.
"""
import sys
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, "/lus/eagle/projects/lighthouse-purdue/harshana/svd_v2")
from utils.misc import DictAsMember
from dataset import create_dataset

CONFIG = "checkpoints/reFlux/ContextProjectorAlign_JEPA_alcf.yml"

c = OmegaConf.load(CONFIG)
data = yaml.safe_load(OmegaConf.to_yaml(c, resolve=True))
opt = DictAsMember(**data)

for split_name in ["train", "val"]:
    print(f"=== warming {split_name} ===", flush=True)
    dopt = opt.datasets[split_name]
    ds = create_dataset(dopt)
    print(f"{split_name}: len={len(ds)}", flush=True)

print("WARM_DONE", flush=True)

"""Quick smoke test for Flux2KleinHybrid_model: instantiate, run a handful of
optimize_parameters() steps, and print losses — bypassing trackers and the
full BaseModel.train() loop so failures surface fast without spinning up
comet/checkpointing machinery.
"""
import logging

import torch
import yaml
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from models import create_model
from utils.misc import DictAsMember

logging.basicConfig(level=logging.INFO)
logger = get_logger("smoke")

N_STEPS = 3


def main():
    with open("checkpoints/reFlux/Flux2KleinHybrid.yml") as f:
        data = OmegaConf.load(f)
        data = OmegaConf.to_yaml(data, resolve=True)
        data = yaml.safe_load(data)
    opt = DictAsMember(**data)
    opt.name = "test"
    opt.tracker_project_name = "test"
    opt.report_to = "tensorboard"
    opt.comment = ""
    opt.test = True
    opt.opt_path = "checkpoints/reFlux/Flux2KleinHybrid.yml"

    model = create_model(opt, logger)
    model.setup_dataloaders()
    model.prepare()

    for m in model.models:
        m.train()

    step = 0
    for batch in model.dataloader:
        model.feed_data(batch, is_train=True)
        losses = model.optimize_parameters()
        printable = {k: (v.item() if torch.is_tensor(v) else v) for k, v in losses.items()}
        print(f"STEP {step}: {printable}", flush=True)
        for k, v in printable.items():
            assert v == v, f"NaN in loss {k} at step {step}"  # noqa: PLR0124 (NaN check)
        step += 1
        if step >= N_STEPS:
            break

    print("SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()

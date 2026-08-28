"""Verify the base_model.py load_model_hook fix: resume from the K1 Hybrid
checkpoint, run one real training step, and confirm the LoRA weights actually
change (they didn't, before the fix -- see peft_resume_optimizer_disconnect
memory note)."""
import logging

import torch
import yaml
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from models import create_model
from utils.misc import DictAsMember

logging.basicConfig(level=logging.INFO)
logger = get_logger("verify")


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
    # Force a real resume from the K1 checkpoint, matching the actual
    # trainer.py resume path (accelerator.load_state + after_resume_checkpoint).
    opt.path.resume_from_path = "/scratch/gilbreth/wweligam/experiments/Motion/Flux2KleinHybrid_2026_07_29_02_30_28"
    opt.path.resume_from_checkpoint = "checkpoint-20000"

    model = create_model(opt, logger)
    model.setup_dataloaders()
    model.prepare()  # builds fresh optimizers BEFORE the resume, same as train()

    gen_param = model.optimizers[0].param_groups[0]["params"][0]
    before = gen_param.detach().clone()
    print(f"tracked param id before resume: {id(gen_param)}")

    import os
    load_path = os.path.join(opt.path.resume_from_path, opt.path.resume_from_checkpoint)
    model.accelerator.print(f"Resuming from checkpoint {load_path}")
    model.accelerator.load_state(load_path)
    model._sanitize_optimizer_state_shapes()
    model.load_other_parameters(load_path)
    model.after_resume_checkpoint(load_path)

    gen_param_after_resume = model.optimizers[0].param_groups[0]["params"][0]
    print(f"tracked param id after resume: {id(gen_param_after_resume)}")
    print(f"same object as before resume: {gen_param_after_resume is gen_param}")
    print(f"value changed by resume (loading real weights, expected): "
          f"{not torch.equal(gen_param_after_resume.detach(), before)}")

    pre_step = gen_param_after_resume.detach().clone()

    for m in model.models:
        m.train()

    batch = next(iter(model.dataloader))
    model.feed_data(batch, is_train=True)
    losses = model.optimize_parameters()
    print("STEP losses:", {k: (v.item() if torch.is_tensor(v) else v) for k, v in losses.items()})

    post_step = model.optimizers[0].param_groups[0]["params"][0].detach().clone()
    changed = not torch.equal(pre_step, post_step)
    max_diff = (pre_step.float() - post_step.float()).abs().max().item()
    print(f"\nPARAM CHANGED AFTER ONE OPTIMIZER STEP POST-RESUME: {changed}")
    print(f"max abs diff: {max_diff}")
    if changed:
        print("FIX CONFIRMED WORKING")
    else:
        print("FIX DID NOT WORK -- optimizer is still disconnected")


if __name__ == "__main__":
    main()

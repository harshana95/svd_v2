"""D-JEPA baseline: loads the official pretrained ImageNet generator and runs
inference on it as a training-free restoration baseline.

D-JEPA (https://arxiv.org/abs/2410.03755, ICLR 2025) was never trained to
deblur/restore images -- its checkpoints are ImageNet class-/unconditional
generators. `djepa_wrapper.DJEPARestorer` adapts it via an SDEdit-style
partial-regeneration trick (see that file for details). This class has no
trainable parameters; it exists to plug that wrapper into this repo's
BaseModel/dataloader/logging/metrics machinery so it can be evaluated the
same way as every other model here.

Run with `-infer` (`is_train: false`) so `trainer.py` skips straight to
`validation()`:

    python trainer.py -opt checkpoints/Baselines/DJEPA_base.yml -infer

See checkpoints/Baselines/DJEPA_base.yml for the full set of config knobs.
"""
import einops
import numpy as np
import torch

from models.base_model import BaseModel
from utils import log_image, log_metrics

from .djepa_wrapper import DJEPARestorer, default_cache_dir


class DJEPA_model(BaseModel):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        variant = opt.get("djepa_variant", "base")
        img_size = int(opt.get("djepa_img_size", 256))
        cache_dir = opt.get("djepa_cache_dir", None) or default_cache_dir()

        logger.info(f"Loading D-JEPA-{variant} (img_size={img_size}) ...")
        self.restorer = DJEPARestorer(
            variant=variant,
            img_size=img_size,
            checkpoint_path=opt.get("djepa_checkpoint_path", None),
            vae_ckpt_path=opt.get("djepa_vae_ckpt_path", None),
            cache_dir=cache_dir,
            use_ema=bool(opt.get("djepa_use_ema", True)),
            num_sampling_steps=str(opt.get("djepa_num_sampling_steps", "100")),
            logger=logger,
        )
        self.restorer.model.to(self.accelerator.device, dtype=weight_dtype)
        self.restorer.vae.to(self.accelerator.device, dtype=weight_dtype)

        # No trainable parameters -- self.models stays empty, so
        # BaseModel.prepare()/train() skip optimizer setup and checkpointing
        # cleanly. This baseline is meant to be run with `-infer`.
        if self.is_train:
            logger.warning(
                "DJEPA_model has no trainable parameters; is_train=true will just run "
                "validation with nothing to optimize. Use `-infer` (is_train: false) instead."
            )

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    def optimize_parameters(self):
        raise NotImplementedError(
            "DJEPA_model is a frozen inference-only baseline with nothing to train. "
            "Run it with `-infer` (is_train: false) instead."
        )

    @torch.no_grad()
    def forwardpass(self, lq_image: torch.Tensor) -> torch.Tensor:
        if lq_image.shape[1] == 1:
            lq_image = einops.repeat(lq_image, "b 1 h w -> b 3 h w")
        val = self.opt.val
        out = self.restorer.restore(
            lq_image.to(self.accelerator.device, dtype=self.weight_dtype).clip(-1, 1),
            strength=float(val.get("djepa_strength", 0.6)),
            num_iter=int(val.get("djepa_num_iter", 64)),
            cfg_scale=float(val.get("djepa_cfg_scale", 1.0)),
            cfg_schedule=val.get("djepa_cfg_schedule", "linear"),
            temperature=float(val.get("djepa_temperature", 1.0)),
            time_shifting_factor=float(val.get("djepa_time_shifting_factor", 1.0)),
            class_label=val.get("djepa_class_label", None),
            vae_encode_mode=val.get("djepa_vae_encode_mode", "mode"),
            progress=bool(val.get("djepa_progress", False)),
        )
        return out.float()

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        bsz = lq.shape[0]

        out = self.forwardpass(lq)

        if lq.shape[1] == 1:
            lq = einops.repeat(lq, "b 1 h w -> b 3 h w")
        if gt.shape[1] == 1:
            gt = einops.repeat(gt, "b 1 h w -> b 3 h w")

        lq_np = (lq.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
        gt_np = (gt.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        for i in range(bsz):
            idx += 1
            images = np.clip(np.stack([lq_np[i], gt_np[i], out_np[i]]), 0, 1)
            log_image(self.opt, self.accelerator, images, f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator, out_np[i:i + 1], f"out_{idx:04d}", self.global_step)
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

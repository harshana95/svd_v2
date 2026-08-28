"""
TrajectoryINR_model: trains models/archs/TrajectoryINR_arch.py by
analysis-by-synthesis.

Given a blurry observation y, the network predicts a candidate sharp image
x_hat and a continuous SE(3) camera trajectory theta(t) (see the arch
module's docstring for the full theory). Re-blurring x_hat along theta(t)
should reproduce y -- that reconstruction is the primary (self-supervised)
training signal, so the model can in principle train on blurry images alone.
When ground-truth sharp images are available (as they are from
dataset/synth_motionblur_dataset.py::SyntheticMotionBlurDataset) a direct
supervised term on x_hat is added, and a smoothness prior on theta(t)
operationalizes the "restricting the trajectory's parameter space reduces
the ill-posedness of the inverse problem" argument -- something a free-form
dense kernel has no equivalent of.

See models/ContinuousTrajectory/README.md for the full write-up, run
instructions, and open extension points.
"""
import numpy as np
import torch

from models.archs import define_network
from models.base_model import BaseModel
from utils.loss import Loss
from utils import log_image, log_metrics


def trajectory_smoothness_loss(rotvec, trans, t):
    """Penalizes the discrete second derivative (acceleration) of theta(t)
    along the sorted time samples -- a low-curvature-path prior. rotvec,
    trans: (B, N, 3).  t: (N,), sorted increasing."""
    if rotvec.shape[1] < 3:
        return rotvec.new_zeros(())
    pose = torch.cat([rotvec, trans], dim=-1)  # (B, N, 6)
    dt = (t[1:] - t[:-1]).clamp_min(1e-6)
    vel = (pose[:, 1:] - pose[:, :-1]) / dt.view(1, -1, 1)
    dt_mid = ((dt[1:] + dt[:-1]) * 0.5).clamp_min(1e-6)
    accel = (vel[:, 1:] - vel[:, :-1]) / dt_mid.view(1, -1, 1)
    return accel.pow(2).mean()


class TrajectoryINR_model(BaseModel):

    def __init__(self, opt, logger):
        super(TrajectoryINR_model, self).__init__(opt, logger)

        if opt.train.get('patched', False):
            # The blur-synthesis warp pulls pixel content from outside any
            # local crop, so patched training/inference isn't meaningful here.
            raise NotImplementedError(
                "TrajectoryINR_model needs the full image to synthesize blur; "
                "set train.patched (and val.patched) to false.")

        self.net_g = define_network(opt.network)
        self.models.append(self.net_g)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt.train
        # train.loss supervises the recovered sharp image x_hat (only used
        # for batches where ground truth is available); reblur_loss
        # supervises the re-synthesized blur against the observed input and
        # is always available, even without ground truth.
        self.criterion_sharp = Loss(train_opt.get('loss', '1*L1')).to(self.accelerator.device)
        self.criterion_reblur = Loss(train_opt.get('reblur_loss', '1*L1')).to(self.accelerator.device)
        self.lambda_sharp = train_opt.get('lambda_sharp', 1.0)
        self.lambda_reblur = train_opt.get('lambda_reblur', 1.0)
        self.lambda_smooth = train_opt.get('lambda_smooth', 0.01)
        self.num_time_samples_val = train_opt.get('num_time_samples_val', 64)

    def feed_data(self, data, is_train=True):
        self.sample = data

    def forwardpass(self, lq, num_time_samples=None, t=None):
        return self.net_g(lq, num_time_samples=num_time_samples, t=t)

    def optimize_parameters(self):
        lq_key = self.dataloader.dataset.lq_key
        gt_key = self.dataloader.dataset.gt_key
        lq = self.sample[lq_key]
        gt = self.sample.get(gt_key, None)

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        out = self.forwardpass(lq)

        losses = {}
        reblur_losses = self.criterion_reblur(out['blur'], lq)
        loss_total = self.lambda_reblur * reblur_losses['all']
        losses['reblur'] = reblur_losses['all']

        if gt is not None and self.lambda_sharp > 0:
            sharp_losses = self.criterion_sharp(out['sharp'], gt)
            loss_total = loss_total + self.lambda_sharp * sharp_losses['all']
            losses['sharp'] = sharp_losses['all']

        smooth = trajectory_smoothness_loss(out['rotvec'], out['trans'], out['t'])
        loss_total = loss_total + self.lambda_smooth * smooth
        losses['traj_smooth'] = smooth
        losses['all'] = loss_total

        self.accelerator.backward(loss_total)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt.train.optim.get('max_grad_norm', 1.0))
        for optimizer in self.optimizers:
            optimizer.step()

        return losses

    @torch.no_grad()
    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        lq = self.sample[lq_key]
        gt = self.sample.get(gt_key, None)

        out = self.forwardpass(lq, num_time_samples=self.num_time_samples_val)
        self.calculate_flops(lq)

        lq_np = lq.cpu().numpy() * 0.5 + 0.5
        sharp_np = out['sharp'].cpu().numpy() * 0.5 + 0.5
        reblur_np = out['blur'].cpu().numpy() * 0.5 + 0.5
        gt_np = gt.cpu().numpy() * 0.5 + 0.5 if gt is not None else None

        for i in range(len(lq_np)):
            idx += 1
            log_image(self.opt, self.accelerator, np.clip(np.stack([lq_np[i]]), 0, 1), f'lq_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([sharp_np[i]]), 0, 1), f'sharp_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([reblur_np[i]]), 0, 1), f'reblur_{idx:04d}', self.global_step)
            # Reblur-consistency metric is always available (self-supervised
            # signal, no ground truth needed).
            log_metrics(lq_np[i], reblur_np[i], self.opt.val.metrics, self.accelerator, self.global_step, comment='_reblur')
            if gt_np is not None:
                log_image(self.opt, self.accelerator, np.clip(np.stack([gt_np[i]]), 0, 1), f'gt_{idx:04d}', self.global_step)
                log_metrics(gt_np[i], sharp_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

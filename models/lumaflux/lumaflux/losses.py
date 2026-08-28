"""Training objective -- paper Eq. (18).

    L = lambda1 * || Y_lin - Y*_lin ||_1
      + lambda2 * || x_lin - x*_lin ||_1
      + lambda3 * L_spline-smooth

Linear terms use the inverse PQ EOTF (the model output and reference are PQ /
BT.2020 signals in [0, 1]).  ``L_spline-smooth`` (adjacent RQS knot slope changes)
is supplied by the RQS decoder.

An optional flow-matching velocity term (``fm_weight``) is provided for training
stability; it defaults to 0 to match the paper's objective exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import pq_eotf, luma2020


class LumaFluxLoss(nn.Module):
    def __init__(self, lambda1: float = 5.0, lambda2: float = 1.0,
                 lambda3: float = 0.1, fm_weight: float = 0.0):
        super().__init__()
        self.l1 = lambda1
        self.l2 = lambda2
        self.l3 = lambda3
        self.fm = fm_weight

    def forward(self, x_hat_pq: torch.Tensor, x_ref_pq: torch.Tensor,
                spline_smooth: torch.Tensor,
                v_pred: torch.Tensor | None = None,
                v_target: torch.Tensor | None = None):
        x_hat_lin = pq_eotf(x_hat_pq)
        x_ref_lin = pq_eotf(x_ref_pq)
        y_hat = luma2020(x_hat_lin)
        y_ref = luma2020(x_ref_lin)

        loss_y = F.l1_loss(y_hat, y_ref)
        loss_x = F.l1_loss(x_hat_lin, x_ref_lin)
        loss = self.l1 * loss_y + self.l2 * loss_x + self.l3 * spline_smooth

        logs = {
            "loss_y": loss_y.detach(),
            "loss_x": loss_x.detach(),
            "loss_spline": spline_smooth.detach(),
        }
        if self.fm > 0.0 and v_pred is not None and v_target is not None:
            loss_fm = F.mse_loss(v_pred.float(), v_target.float())
            loss = loss + self.fm * loss_fm
            logs["loss_fm"] = loss_fm.detach()

        logs["loss"] = loss.detach()
        return loss, logs

import torch
from torch import Tensor
from typing import Optional
from torch import nn
from typing import Final


class DfAlphaLoss(nn.Module):
    """Add a penalty to use DF for very noisy segments.

    Starting from lsnr_thresh, the penalty is increased and has its maximum at lsnr_min.
    """
    factor: 'Final[float]'
    lsnr_thresh: 'Final[float]'
    lsnr_min: 'Final[float]'

    def __init__(self, factor: 'float'=1, lsnr_thresh: 'float'=-7.5,
        lsnr_min: 'float'=-10.0):
        super().__init__()
        self.factor = factor
        self.lsnr_thresh = lsnr_thresh
        self.lsnr_min = lsnr_min

    def forward(self, pred_alpha: 'Tensor', target_lsnr: 'Tensor'):
        w = self.lsnr_mapping(target_lsnr, self.lsnr_thresh, self.lsnr_min
            ).view_as(pred_alpha)
        l_off = (pred_alpha * w).square().mean()
        w = self.lsnr_mapping(target_lsnr, self.lsnr_thresh + 2.5, 0.0
            ).view_as(pred_alpha)
        l_on = 0.1 * ((1 - pred_alpha) * w).abs().mean()
        return l_off + l_on

    def lsnr_mapping(self, lsnr: 'Tensor', lsnr_thresh: 'float', lsnr_min:
        'Optional[float]'=None) ->Tensor:
        """Map lsnr_min to 1 and lsnr_thresh to 0"""
        lsnr_min = float(self.lsnr_min) if lsnr_min is None else lsnr_min
        a_ = 1 / (lsnr_thresh - lsnr_min)
        b_ = -a_ * lsnr_min
        return 1 - torch.clamp(a_ * lsnr + b_, 0.0, 1.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

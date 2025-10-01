import torch
import torch.nn as nn
import torch.utils.data


class ModifiedSmoothedL1(nn.Module):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                     |x| - 0.5 / sigma^2,    otherwise
    """

    def __init__(self, sigma):
        super(ModifiedSmoothedL1, self).__init__()
        self.sigma2 = sigma * sigma

    def forward(self, deltas, targets, sigma=None):
        sigma2 = self.sigma2 if sigma is None else sigma * sigma
        diffs = deltas - targets
        option1 = diffs * diffs * 0.5 * sigma2
        option2 = torch.abs(diffs) - 0.5 / sigma2
        condition_for_1 = (diffs < 1.0 / sigma2).float()
        smooth_l1 = option1 * condition_for_1 + option2 * (1 - condition_for_1)
        return smooth_l1


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sigma': 4}]

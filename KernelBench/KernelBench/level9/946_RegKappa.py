import torch
from torch.nn.modules.loss import _Loss
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class RegKappa(_Loss):

    def __init__(self, ignore_index=None):
        super(RegKappa, self).__init__()
        self.min = min
        self.max = max
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]
        target = target.float()
        num = 2 * torch.sum(input * target)
        denom = input.norm(2) + target.norm(2)
        eps = 1e-07
        kappa = num / (denom + eps)
        return 1.0 - kappa


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

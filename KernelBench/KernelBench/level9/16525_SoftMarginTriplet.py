import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SoftMarginTriplet(_Loss):
    __constants__ = ['reduction']
    """
    inputs `x1`, `x2`, two 1D mini-batch `Tensor`s,
    and a label 1D mini-batch tensor `y` with values (`1` or `-1`).

    If `y == 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for `y == -1`.

    The loss function for each sample in the mini-batch is:
    
    loss(x, y) = max(0, -y * (x1 - x2) + margin)  
    
    reduction='elementwise_mean'|'none'|'sum'
    """

    def __init__(self, margin=0.0, size_average=None, reduce=None,
        reduction='elementwise_mean'):
        super(SoftMarginTriplet, self).__init__(size_average, reduce, reduction
            )
        self.margin = margin

    def forward(self, dist_ap, dist_an, softmargin):
        loss = F.relu(dist_ap - dist_an + softmargin * self.margin)
        if self.reduction == 'elementwise_mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

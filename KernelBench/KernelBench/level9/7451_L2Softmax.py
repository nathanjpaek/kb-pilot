import math
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


class L2Softmax(_WeightedLoss):
    """L2Softmax from
    `"L2-constrained Softmax Loss for Discriminative Face Verification"
    <https://arxiv.org/abs/1703.09507>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    alpha: float.
        The scaling parameter, a hypersphere with small alpha
        will limit surface area for embedding features.
    p: float, default is 0.9.
        The expected average softmax probability for correctly
        classifying a feature.
    from_normx: bool, default is False.
         Whether input has already been normalized.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, alpha, p=0.9, from_normx=False, weight=None,
        size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(L2Softmax, self).__init__(weight, size_average, reduce, reduction
            )
        alpha_low = math.log(p * (classes - 2) / (1 - p))
        assert alpha > alpha_low, 'For given probability of p={}, alpha should higher than {}.'.format(
            p, alpha_low)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.from_normx = from_normx

    def forward(self, x, target):
        if not self.from_normx:
            x = F.normalize(x, 2, dim=-1)
        x = x * self.alpha
        return F.cross_entropy(x, target, weight=self.weight, ignore_index=
            self.ignore_index, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'classes': 4, 'alpha': 4}]

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import *


class Criterion(_Loss):

    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        return


class CeCriterion(Criterion):

    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        """
        交叉熵损失
        :param alpha:
        :type alpha:
        :param name:
        :type name:
        """
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False,
                ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

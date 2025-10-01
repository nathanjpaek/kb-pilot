import torch
from torch import Tensor
from torch.backends import cudnn as cudnn
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from typing import List


class SoftCrossEntropyLoss(nn.Module):
    """Calculate the CrossEntropyLoss with soft targets.

    :param weight: Weight to assign to each of the classes. Default: None
    :type weight: list of float
    :param reduction: The way to reduce the losses: 'none' | 'mean' | 'sum'.
        'none': no reduction,
        'mean': the mean of the losses,
        'sum': the sum of the losses.
    :type reduction: str
    """

    def __init__(self, weight: 'List[float]'=None, reduction: 'str'='mean'):
        super().__init__()
        if weight is None:
            self.weight = None
        else:
            self.register_buffer('weight', torch.Tensor(weight))
        self.reduction = reduction

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        """Calculate the loss.

        :param input: prediction logits
        :param target: target probabilities
        :return: loss
        """
        n, k = input.shape
        losses = input.new_zeros(n)
        for i in range(k):
            cls_idx = input.new_full((n,), i, dtype=torch.long)
            loss = F.cross_entropy(input, cls_idx, reduction='none')
            if self.weight is not None:
                loss = loss * self.weight[i]
            losses += target[:, i].float() * loss
        if self.reduction == 'mean':
            losses = losses.mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction != 'none':
            raise ValueError(f'Unrecognized reduction: {self.reduction}')
        return losses


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
from typing import Optional


def focal_loss(input: 'torch.Tensor', target: 'torch.Tensor', gamma:
    'float'=0, weight: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    """
    Returns the focal loss between `target` and `input`

    :math:`\\text{FL}(p_t)=-(1-p_t)^\\gamma\\log(p_t)`
    """
    if input.shape[1] == 1:
        logp = nn.functional.binary_cross_entropy_with_logits(input, target)
    else:
        logp = nn.functional.cross_entropy(input, target, weight=weight)
    p = torch.exp(-logp)
    loss = (1 - p) ** gamma * logp
    return loss.mean()


class FocalLoss(nn.Module):
    """
    The focal loss

    https://arxiv.org/abs/1708.02002

    See :func:`torchelie.loss.focal_loss` for details.
    """

    def __init__(self, gamma: 'float'=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'
        ) ->torch.Tensor:
        return focal_loss(input, target, self.gamma)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

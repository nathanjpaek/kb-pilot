import torch
from typing import Optional
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel


class ContourDTConsistency(nn.Module):
    """Consistency regularization between the instance contour map and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): contour logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None.
    """

    def forward(self, pred1: 'torch.Tensor', pred2: 'torch.Tensor', mask:
        'Optional[torch.Tensor]'=None):
        contour_prob = torch.sigmoid(pred1)
        distance_abs = torch.abs(torch.tanh(pred2))
        assert contour_prob.shape == distance_abs.shape
        loss = contour_prob * distance_abs
        loss = loss ** 2
        if mask is not None:
            loss *= mask
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

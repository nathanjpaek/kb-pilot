import torch
from typing import Optional
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel


class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.

    Args:
        pred (torch.Tensor): foreground logits.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """

    def forward(self, pred: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None
        ):
        pred = torch.sigmoid(pred)
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=0.01)
        loss = 1.0 / diff
        if mask is not None:
            loss *= mask
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

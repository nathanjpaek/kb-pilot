import torch
from typing import Optional
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class ForegroundDTConsistency(nn.Module):
    """Consistency regularization between the binary foreground mask and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): foreground logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """

    def forward(self, pred1: 'torch.Tensor', pred2: 'torch.Tensor', mask:
        'Optional[torch.Tensor]'=None):
        log_prob_pos = F.logsigmoid(pred1)
        log_prob_neg = F.logsigmoid(-pred1)
        distance = torch.tanh(pred2)
        dist_pos = torch.clamp(distance, min=0.0)
        dist_neg = -torch.clamp(distance, max=0.0)
        loss_pos = -log_prob_pos * dist_pos
        loss_neg = -log_prob_neg * dist_neg
        loss = loss_pos + loss_neg
        if mask is not None:
            loss *= mask
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

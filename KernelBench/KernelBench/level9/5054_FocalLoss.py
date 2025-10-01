import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    from
    https://github.com/CellProfiling/HPA-competition-solutions/blob/master/bestfitting/src/layers/loss.py
    """

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + ((-max_val).exp() + (-
            logit - max_val).exp()).log()
        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

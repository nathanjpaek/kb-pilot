import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109"""

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + ((-max_val).exp() + (-
            logit - max_val).exp()).log()
        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        loss = loss.sum(dim=1) if len(loss.size()) == 2 else loss
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

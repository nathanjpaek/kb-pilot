import torch
from torch import nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        eps = 1e-09
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1).float()
        intersection = torch.sum(m1 * m2, 1)
        union = torch.sum(m1, dim=1) + torch.sum(m2, dim=1)
        score = (2 * intersection + eps) / (union + eps)
        score = (1 - score).mean()
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocUnetLossPow(nn.Module):
    """
    对应公式5的loss
    """

    def __init__(self, r=0.1):
        super(DocUnetLossPow, self).__init__()
        self.r = r

    def forward(self, y, label):
        d = y - label
        lossf = d.pow(2).mean() - self.r * d.mean().pow(2)
        loss = F.mse_loss(y, label) + lossf
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

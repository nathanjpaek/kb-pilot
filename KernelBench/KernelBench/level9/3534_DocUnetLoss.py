import torch
import torch.nn as nn
import torch.nn.functional as F


class DocUnetLoss(nn.Module):
    """
    只使用一个unet的loss 目前使用这个loss训练的比较好
    """

    def __init__(self, r=0.1):
        super(DocUnetLoss, self).__init__()
        self.r = r

    def forward(self, y, label):
        d = y - label
        lossf = torch.abs(d).mean() - self.r * torch.abs(d.mean())
        loss = F.mse_loss(y, label) + lossf
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

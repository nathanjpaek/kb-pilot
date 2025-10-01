import torch
from torch import nn
from torchvision.models import *


class LblLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_batch, true_batch):
        wgt = torch.ones_like(pred_batch)
        wgt[true_batch > 0] = 100
        dis = (pred_batch - true_batch) ** 2
        return (dis * wgt).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

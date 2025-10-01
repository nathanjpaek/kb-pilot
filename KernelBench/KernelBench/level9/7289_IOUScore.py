import torch
from torch import nn
from torch import torch


class IOUScore(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Yp, Yt):
        output_ = Yp > 0.5
        target_ = Yt > 0.5
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = intersection / union
        return iou


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

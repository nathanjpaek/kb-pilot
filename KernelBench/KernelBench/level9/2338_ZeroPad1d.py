import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.utils.data
import torch.onnx.operators
import torch.optim


class ZeroPad1d(nn.Module):

    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pad_left': 4, 'pad_right': 4}]

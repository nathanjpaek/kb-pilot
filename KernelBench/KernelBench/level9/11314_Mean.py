import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import *
import torch.optim.lr_scheduler
import torch.quantization
import torch.onnx
import torch.testing


class Mean(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Mean, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: 'torch.Tensor'):
        return torch.mean(x, *self.args, **self.kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

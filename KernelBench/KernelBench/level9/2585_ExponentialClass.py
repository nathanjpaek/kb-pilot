import torch
import torch.cuda
import torch.distributed
from torch.cuda.amp import autocast as autocast
import torch.utils.data
import torch.optim


class ExponentialClass(torch.nn.Module):

    def __init__(self):
        super(ExponentialClass, self).__init__()

    def forward(self, x):
        return torch.exp(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

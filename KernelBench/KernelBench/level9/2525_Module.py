from torch.nn import Module
import torch
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class Module(torch.nn.Module):

    def __init__(self):
        super(Module, self).__init__()
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)

    def forward(self, x):
        y = self.conv(x)
        return y


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

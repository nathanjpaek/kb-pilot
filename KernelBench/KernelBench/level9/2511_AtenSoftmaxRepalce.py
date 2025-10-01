import torch
import torch.nn as nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class AtenSoftmaxRepalce(nn.Module):

    def __init__(self, dim=-1):
        super(AtenSoftmaxRepalce, self).__init__()
        self.softmax = torch.nn.Softmax(dim)

    def forward(self, x):
        return self.softmax(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

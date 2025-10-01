import torch
from torch import nn
import torch.nn.functional as F
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.cuda
import torch.backends.quantized


class ConvRelu(nn.Module):

    def __init__(self):
        super(ConvRelu, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=
            (4, 2), dilation=(3, 1))

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


def get_inputs():
    return [torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {}]

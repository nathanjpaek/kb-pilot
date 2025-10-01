import torch
from torch import nn
import torch.nn.functional as F
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.cuda
import torch.backends.quantized


class Conv2dRelu_Fixed(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]

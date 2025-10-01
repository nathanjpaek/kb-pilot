import torch
from torch import nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.cuda
import torch.backends.quantized


class ConvSigmoidInplace(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, image_size):
        super(ConvSigmoidInplace, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
            image_size)

    def forward(self, x):
        a = self.conv2d(x)
        b = torch.sigmoid_(a)
        c = torch.add(b, b)
        return c


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'image_size': 4}]

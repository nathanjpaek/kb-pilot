import math
import torch
import torch.jit
import torch.nn as nn
import torch.nn.init as init
import torch.onnx


def _initialize_orthogonal(conv):
    prelu_gain = math.sqrt(2)
    init.orthogonal(conv.weight, gain=prelu_gain)
    if conv.bias is not None:
        conv.bias.data.zero_()


class UpscaleBlock(nn.Module):

    def __init__(self, n_filters):
        super(UpscaleBlock, self).__init__()
        self.upscaling_conv = nn.Conv2d(n_filters, 4 * n_filters,
            kernel_size=3, padding=1)
        self.upscaling_shuffler = nn.PixelShuffle(2)
        self.upscaling = nn.PReLU(n_filters)
        _initialize_orthogonal(self.upscaling_conv)

    def forward(self, x):
        return self.upscaling(self.upscaling_shuffler(self.upscaling_conv(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_filters': 4}]

import torch
from torch import nn


class SeparableConv2d(nn.Module):
    """Implements a depthwise separable 2D convolution as described
    in MobileNet (https://arxiv.org/abs/1704.04861)
    
    See: [SeparableConv2D in Keras](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv2D)
    
    Implementation due to https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
            padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]

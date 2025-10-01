import torch
import torch.nn as nn
import torch.nn.parallel
import torch.onnx


def conv1x1(in_planes, out_planes, bias=False):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
        padding=0, bias=bias)


class SpatialAttentionModule(nn.Module):

    def __init__(self, input_size, input_channels):
        super(SpatialAttentionModule, self).__init__()
        self.avgPool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.hidden_size = input_size // 2 - 1
        self.upsampling = nn.Upsample(size=(input_size, input_size), mode=
            'bilinear')
        self.conv_f = conv1x1(input_channels, input_channels)
        self.conv_g = conv1x1(input_channels, input_channels)
        self.conv_h = conv1x1(input_channels, input_channels)
        self.softmax = torch.nn.Softmax()

    def forward(self, spatial_features):
        downsampled = self.maxPool(self.avgPool(spatial_features))
        f = self.conv_f(downsampled)
        g = self.conv_g(downsampled)
        h = self.conv_h(downsampled)
        f_transposed = torch.transpose(f, 2, 3)
        attention_map = self.softmax(f_transposed * g)
        out = self.upsampling(h * attention_map)
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'input_size': 4, 'input_channels': 4}]

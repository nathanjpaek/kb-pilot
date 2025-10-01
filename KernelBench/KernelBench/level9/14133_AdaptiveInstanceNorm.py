import math
import torch
import torch.nn as nn
import torch.utils.cpp_extension


@torch.no_grad()
def scaling_init(tensor, scale=1, dist='u'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale /= (fan_in + fan_out) / 2
    if dist == 'n':
        std = math.sqrt(scale)
        return tensor.normal_(0.0, std)
    elif dist == 'u':
        bound = math.sqrt(3 * scale)
        return tensor.uniform_(-bound, bound)


def linear(in_features, out_features, scale=1.0):
    _linear = nn.Linear(in_features, out_features)
    scaling_init(_linear.weight, scale)
    nn.init.zeros_(_linear.bias)
    return _linear


class EqualizedLR(nn.Module):
    """
    equalized learning rate
    """

    def __init__(self, layer, gain=2):
        super(EqualizedLR, self).__init__()
        self.wscale = (gain / layer.weight[0].numel()) ** 0.5
        self.layer = layer

    def forward(self, x, gain=2):
        x = self.layer(x * self.wscale)
        return x


class EqualizedLinear(nn.Module):
    """
    equalized fully connected layer
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        linear = nn.Linear(in_channels, out_channels)
        linear.weight.data.normal_(0, 1)
        linear.bias.data.fill_(0)
        self.linear = EqualizedLR(linear)

    def forward(self, x):
        x = self.linear(x)
        return x


class AdaptiveInstanceNorm(nn.Module):
    """
    AdaIN
    """

    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, eps=1e-08)
        self.linear = EqualizedLinear(style_dim, channels * 2)
        self.linear.linear.layer.bias.data[:channels] = 1.0

    def forward(self, x, style):
        norm = self.norm(x)
        style = self.linear(style).unsqueeze(2).unsqueeze(3)
        ys, yb = style.chunk(2, 1)
        x = ys * norm + yb
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'style_dim': 4}]

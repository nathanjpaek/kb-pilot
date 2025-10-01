import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8, min_value=None):
    """
    The channel number of each layer should be divisable by 8.
    The function is taken from
    github.com/rwightman/pytorch-image-models/master/timm/models/layers/helpers.py
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels: 'int', reduction: 'int'=4, out_channels:
        'int'=-1, **kwargs: dict):
        super(SqueezeExcitation, self).__init__()
        assert in_channels > 0
        num_reduced_channels = make_divisible(max(out_channels, 8) //
            reduction, 8)
        self.fc1 = nn.Conv2d(in_channels, num_reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(num_reduced_channels, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = F.adaptive_avg_pool2d(inp, 1)
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x + inp


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]

import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class JustConvBody(nn.Module):

    def __init__(self, in_channels=4):
        super(JustConvBody, self).__init__()
        self.feature_dim = 7 * 7 * 64
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8,
            stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]

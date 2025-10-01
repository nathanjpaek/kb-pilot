import torch
import torch.nn as nn


class AdaIN2d(nn.Module):

    def __init__(self, in_channels, in_features):
        super(AdaIN2d, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False,
            track_running_stats=False)
        self.net = nn.Linear(in_features, 2 * in_channels)
        self.reset_parameters()

    def forward(self, x, h):
        h = self.net(h)
        bs, fs = h.size()
        h.view(bs, fs, 1, 1)
        b, s = h.chunk(2, 1)
        x = self.norm(x)
        return x * (s + 1) + b

    def reset_parameters(self):
        nn.init.constant_(self.net.weight, 0.0)
        nn.init.constant_(self.net.bias, 0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'in_features': 4}]

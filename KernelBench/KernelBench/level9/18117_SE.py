import torch
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se1 = nn.Conv2d(w_in, w_se, kernel_size=1, bias=True)
        self.reluse = nn.ReLU(inplace=True)
        self.se2 = nn.Conv2d(w_se, w_in, kernel_size=1, bias=True)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.se1(out)
        out = self.reluse(out)
        out = self.se2(out)
        out = self.sm(out)
        out = x * out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'w_in': 4, 'w_se': 4}]

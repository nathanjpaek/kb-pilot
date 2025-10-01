import torch
import torch.nn as nn


class LCTGate(nn.Module):

    def __init__(self, channels, groups=16):
        super(LCTGate, self).__init__()
        assert channels > 0
        assert groups > 0
        while channels % groups != 0:
            groups //= 2
        self.gn = nn.GroupNorm(groups, channels, affine=True)
        nn.init.ones_(self.gn.bias)
        nn.init.zeros_(self.gn.weight)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.gn(x)
        x = self.gate_activation(x)
        return input * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]

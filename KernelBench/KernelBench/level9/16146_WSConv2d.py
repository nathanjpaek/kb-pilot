import torch
from torchvision.transforms import functional as F
from torch import nn
from torch.nn import functional as F


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, eps=1e-05):
        super().__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(out_channels))
        self.eps = eps ** 2
        fan_in = torch.numel(self.weight[0])
        self.scale = fan_in ** -0.5
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')

    def forward(self, input):
        weight = F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        gain = self.gain.view([-1] + [1] * (weight.ndim - 1))
        weight = gain * self.scale * weight
        return F.conv2d(input, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]

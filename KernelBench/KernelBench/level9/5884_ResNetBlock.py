from torch.nn import Module
import torch
import torch.onnx
from torch.nn import Conv2d
from torch.nn import InstanceNorm2d
from torch.nn.init import kaiming_normal_
from torch.nn.init import xavier_normal_
from torch import relu


def create_init_function(method: 'str'='none'):

    def init(module: 'Module'):
        if method == 'none':
            return module
        elif method == 'he':
            kaiming_normal_(module.weight)
            return module
        elif method == 'xavier':
            xavier_normal_(module.weight)
            return module
        else:
            raise ('Invalid initialization method %s' % method)
    return init


def Conv3(in_channels: 'int', out_channels: 'int', initialization_method='he'
    ) ->Module:
    init = create_init_function(initialization_method)
    return init(Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=False))


class ResNetBlock(Module):

    def __init__(self, num_channels: 'int', initialization_method: 'str'='he'):
        super().__init__()
        self.conv1 = Conv3(num_channels, num_channels, initialization_method)
        self.norm1 = InstanceNorm2d(num_features=num_channels, affine=True)
        self.conv2 = Conv3(num_channels, num_channels, initialization_method)
        self.norm2 = InstanceNorm2d(num_features=num_channels, affine=True)

    def forward(self, x):
        return x + self.norm2(self.conv2(relu(self.norm1(self.conv1(x)))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]

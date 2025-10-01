import torch
import torch.nn as nn


class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.
    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """

    def __init__(self, in_features, growth_rate, drop_rate=0.0, bn_size=8,
        bottleneck=False):
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
                growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(p=drop_rate))

    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlockInput(nn.Sequential):
    """For input dense block, feature map size the same as input"""

    def __init__(self, num_layers, in_features, init_features, growth_rate,
        drop_rate, bn_size=4, bottleneck=False):
        super(_DenseBlockInput, self).__init__()
        self.num_layers = num_layers
        self.add_module('in_conv', nn.Conv2d(in_features, init_features - 1,
            kernel_size=3, stride=1, padding=1))
        for i in range(num_layers - 1):
            layer = _DenseLayer(init_features + i * growth_rate,
                growth_rate, drop_rate=drop_rate, bn_size=bn_size,
                bottleneck=bottleneck)
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, x):
        out = self.in_conv(x)
        out = torch.cat((x, out), 1)
        for i in range(self.num_layers - 1):
            out = self[i + 1](out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_layers': 1, 'in_features': 4, 'init_features': 4,
        'growth_rate': 4, 'drop_rate': 0.5}]

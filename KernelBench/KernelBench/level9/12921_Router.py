import torch
import warnings
import torch.nn as nn


class Router(nn.Module):
    """Convolution + Relu + Global Average Pooling + Sigmoid"""

    def __init__(self, input_nc, input_width, input_height, kernel_size=28,
        soft_decision=True, stochastic=False, **kwargs):
        super(Router, self).__init__()
        self.soft_decision = soft_decision
        self.stochastic = stochastic
        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)
        self.conv1 = nn.Conv2d(input_nc, 1, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = x.mean(dim=-1).mean(dim=-1).squeeze()
        x = self.output_controller(x)
        return x

    def output_controller(self, x):
        if self.soft_decision:
            return self.sigmoid(x)
        if self.stochastic:
            x = self.sigmoid(x)
            return ops.ST_StochasticIndicator()(x)
        else:
            x = self.sigmoid(x)
            return ops.ST_Indicator()(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nc': 4, 'input_width': 4, 'input_height': 4}]

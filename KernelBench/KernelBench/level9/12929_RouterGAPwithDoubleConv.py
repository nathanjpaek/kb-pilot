import torch
import warnings
import torch.nn as nn


class RouterGAPwithDoubleConv(nn.Module):
    """ 2 x (Convolution + Relu) + Global Average Pooling + FC + Sigmoid """

    def __init__(self, input_nc, input_width, input_height, ngf=32,
        kernel_size=3, soft_decision=True, stochastic=False, **kwargs):
        super(RouterGAPwithDoubleConv, self).__init__()
        self.ngf = ngf
        self.soft_decision = soft_decision
        self.stochastic = stochastic
        if max(input_width, input_height) < kernel_size:
            warnings.warn('Router kernel too large, shrink it')
            kernel_size = max(input_width, input_height)
            if max(input_width, input_height) % 2 == 0:
                kernel_size += 1
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=kernel_size,
            padding=padding)
        self.conv2 = nn.Conv2d(ngf, ngf, kernel_size=kernel_size, padding=
            padding)
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(ngf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = out.mean(dim=-1).mean(dim=-1).squeeze()
        out = self.linear1(out).squeeze()
        out = self.output_controller(out)
        return out

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

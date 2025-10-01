import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))


class VdConv1D(nn.Module):
    """
    Conv1D Layer variational dropout

    """

    def __init__(self, in_channels, out_channels, kernel_size, alpha_shape=
        (1, 1), bias=True, stride=1, padding=0, dilation=1):
        super(VdConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha_shape = alpha_shape
        self.groups = 1
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            self.kernel_size))
        self.log_alpha = nn.Parameter(torch.Tensor(*alpha_shape))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.out_bias = lambda input, kernel: F.conv1d(input, kernel, self.
            bias, self.stride, self.padding, self.dilation, self.groups)
        self.out_nobias = lambda input, kernel: F.conv1d(input, kernel,
            None, self.stride, self.padding, self.dilation, self.groups)
        self.reset_parameters()
        self.kl_value = calculate_kl

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x, sample=False):
        mean = self.out_bias(x, self.weight)
        sigma = torch.exp(self.log_alpha) * self.weight * self.weight
        std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))
        if self.training or sample:
            epsilon = std.data.new(std.size()).normal_()
        else:
            epsilon = 0.0
        out = mean + std * epsilon
        kl = self.kl_loss()
        return out, kl

    def kl_loss(self):
        return self.weight.nelement() / self.log_alpha.nelement(
            ) * self.kl_value(self.log_alpha)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]

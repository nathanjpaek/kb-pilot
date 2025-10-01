import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class tLN(nn.Module):

    def __init__(self, dimension, eps=1e-08, trainable=True):
        super(tLN, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1, 1),
                requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1, 1),
                requires_grad=False)

    def forward(self, inp):
        inp.size(0)
        mean = torch.sum(inp, 3, keepdim=True) / inp.shape[3]
        std = torch.sqrt(torch.sum((inp - mean) ** 2, 3, keepdim=True) /
            inp.shape[3] + self.eps)
        x = (inp - mean.expand_as(inp)) / std.expand_as(inp)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(
            x).type(x.type())


class CausalConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1
        ), dilation=(1, 1), groups=1, bias=True):
        _pad = int(np.log2((kernel_size[1] - 1) / 2))
        padding_2 = int(2 ** (np.log2(dilation[1]) + _pad))
        self.__padding = (kernel_size[0] - 1) * dilation[0], padding_2
        super(CausalConv2d, self).__init__(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=self.__padding,
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv2d, self).forward(input)
        if self.__padding[0] != 0:
            return result[:, :, :-self.__padding[0]]
        return result


class DepthConv2d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, dilation=(1, 
        1), stride=(1, 1), padding=(0, 0), causal=False):
        super(DepthConv2d, self).__init__()
        self.padding = padding
        self.linear = nn.Conv2d(input_channel, hidden_channel, (1, 1))
        if causal:
            self.conv1d = CausalConv2d(hidden_channel, hidden_channel,
                kernel, stride=stride, dilation=dilation)
        else:
            self.conv1d = nn.Conv2d(hidden_channel, hidden_channel, kernel,
                stride=stride, padding=self.padding, dilation=dilation)
        self.BN = nn.Conv2d(hidden_channel, input_channel, (1, 1))
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        self.reg1 = tLN(hidden_channel)
        self.reg2 = tLN(hidden_channel)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.linear(input)))
        output = self.reg2(self.nonlinearity2(self.conv1d(output)))
        output = self.BN(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'hidden_channel': 4, 'kernel': 4}]

import torch
import torch.nn as nn


class DepthConv2d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding,
        dilation=1):
        super(DepthConv2d, self).__init__()
        self.conv2d = nn.Conv2d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv2d = nn.Conv2d(hidden_channel, hidden_channel, kernel,
            dilation=dilation, groups=hidden_channel, padding=self.padding)
        self.res_out = nn.Conv2d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv2d(input)))
        output = self.reg2(self.nonlinearity2(self.dconv2d(output)))
        residual = self.res_out(output)
        return residual


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'hidden_channel': 4, 'kernel': 4,
        'padding': 4}]

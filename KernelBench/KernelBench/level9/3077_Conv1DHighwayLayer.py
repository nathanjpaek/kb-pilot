import torch
import torch.nn as nn


class Conv1DHighwayLayer(nn.Module):

    def __init__(self, inchannels, outchannels, kernelsize, activation=
        'relu', stride=1, bias=-1):
        super(Conv1DHighwayLayer, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.kernelsize = kernelsize
        if activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        self.stride = stride
        self.padding = (self.kernelsize - 1) // 2
        self.conv = nn.Conv1d(self.inchannels, self.outchannels, self.
            kernelsize, stride=self.stride, padding=self.padding)
        self.gate = nn.Conv1d(self.inchannels, self.outchannels, self.
            kernelsize, stride=self.stride, padding=self.padding)
        self.gateact = nn.Sigmoid()
        self.gate.bias.data.fill_(bias)

    def forward(self, x):
        H = self.activation(self.conv(x))
        T = self.gateact(self.gate(x))
        out = H * T + x * (1 - T)
        return out


def get_inputs():
    return [torch.rand([4, 2])]


def get_init_inputs():
    return [[], {'inchannels': 4, 'outchannels': 4, 'kernelsize': 4}]

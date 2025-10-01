import torch
import torch.nn as nn


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=2):
        super().__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation)

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]

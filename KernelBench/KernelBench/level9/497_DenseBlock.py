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


class DenseBlock(nn.Module):
    """卷积后拼接在一起"""

    def __init__(self, in_channels, filters, dilation=2):
        super().__init__()
        self.causal_conv1 = CausalConv1d(in_channels, out_channels=filters,
            dilation=dilation)
        self.causal_conv2 = CausalConv1d(in_channels, out_channels=filters,
            dilation=dilation)

    def forward(self, minibatch):
        tanh = torch.tanh(self.causal_conv1(minibatch))
        sig = torch.sigmoid(self.causal_conv2(minibatch))
        return torch.cat([minibatch, tanh * sig], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'filters': 4}]

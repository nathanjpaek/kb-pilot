import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torch.nn


class cnn_layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, bias=True):
        super(cnn_layer, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        return self.relu(self.conv(input))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]

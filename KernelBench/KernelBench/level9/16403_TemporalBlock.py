import torch
from torch import nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


def TemporalConvLayer(input_channels, output_channels, kernel_size):
    m = nn.Conv1d(in_channels=input_channels, out_channels=output_channels,
        kernel_size=kernel_size)
    nn.init.xavier_normal_(m.weight)
    return m


class TemporalBlock(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size,
        dropout):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 0, 0, kernel_size - 1))
        self.tconv = TemporalConvLayer(input_size, num_channels, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.tconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'num_channels': 4,
        'kernel_size': 4, 'dropout': 0.5}]

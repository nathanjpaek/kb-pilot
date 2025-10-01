import torch
import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, in_channels=10, out_channels=10, kernel_size=5,
        pooling_size=3, padding='valid') ->None:
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=padding)
        self.maxPool = nn.MaxPool1d(pooling_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        activation = x
        x = self.maxPool(x)
        x = self.activation(x)
        return x, activation


def get_inputs():
    return [torch.rand([4, 10, 64])]


def get_init_inputs():
    return [[], {}]

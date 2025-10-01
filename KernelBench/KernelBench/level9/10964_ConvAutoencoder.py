import torch
from torch import nn


class ConvAutoencoder(nn.Module):

    def __init__(self, enc_dim=10, channels=1, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, enc_dim, 7, strides, padding=0)
        self.dropout = nn.Dropout(0.2)
        self.t_conv1 = nn.ConvTranspose1d(enc_dim, 1, 7, strides, padding=0)
        self.t_conv2 = nn.ConvTranspose1d(1, 1, 1, strides, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.t_conv1(x)
        x = self.t_conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]

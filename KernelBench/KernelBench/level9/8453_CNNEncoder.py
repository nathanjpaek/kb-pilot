import torch
import torch.nn as nn
from torch.nn import functional as F


class CNNEncoder(nn.Module):

    def __init__(self, out_channels: 'int', kernel_size: 'tuple'):
        super(CNNEncoder, self).__init__()
        self.cnn_encoder = nn.Conv2d(in_channels=1, out_channels=
            out_channels, kernel_size=kernel_size)

    def forward(self, x: 'torch.Tensor'):
        x = x.unsqueeze(dim=1)
        output = F.relu(self.cnn_encoder(x))
        output = output.mean(dim=2)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channels': 4, 'kernel_size': 4}]

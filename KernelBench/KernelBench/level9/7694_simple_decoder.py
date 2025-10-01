import torch
from torch import nn
import torch.utils
import torch.distributions


class simple_decoder(nn.Module):

    def __init__(self, channels, width, height, dropout):
        super(simple_decoder, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.dec_conv = nn.Conv2d(in_channels=self.channels, out_channels=
            self.channels, kernel_size=5, padding=2)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, context=None):
        net = torch.sigmoid(self.dec_conv(x)) * 256
        return net


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'width': 4, 'height': 4, 'dropout': 0.5}]

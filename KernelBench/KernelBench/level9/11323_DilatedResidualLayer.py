import torch
from torch import nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):

    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding
            =dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dilation': 1, 'in_channels': 4, 'out_channels': 4}]

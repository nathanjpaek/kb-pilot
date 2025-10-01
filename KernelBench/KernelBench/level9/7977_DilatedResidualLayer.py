import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):

    def __init__(self, dilation, input_dim, output_dim):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(input_dim, output_dim, 3, padding=
            dilation, dilation=dilation, padding_mode='replicate')
        self.conv_out = nn.Conv1d(output_dim, output_dim, 1)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_out(out)
        return x + out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dilation': 1, 'input_dim': 4, 'output_dim': 4}]

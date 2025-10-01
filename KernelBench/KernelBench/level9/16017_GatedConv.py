import torch
from torch import nn
import torch.nn.init as init


class GatedConv(nn.Module):
    """GatedConv."""

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        """init."""
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_size, out_channels=2 *
            input_size, kernel_size=(width, 1), stride=(1, 1), padding=(
            width // 2 * (1 - nopad), 0))
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        """forward."""
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]

import torch
import torch.utils.data
import torch.nn.functional as F


class ConvReLUNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, padding=kernel_size // 2)
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout_val = dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        if self.dropout_val > 0.0:
            out = self.dropout(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]

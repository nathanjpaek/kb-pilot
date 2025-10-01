import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class _GLUBlock(nn.Module):

    def __init__(self, n_c_in, n_c_out):
        super(_GLUBlock, self).__init__()
        self.pad = nn.ConstantPad1d((1, 2), 0)
        self.conv_data = nn.Conv1d(n_c_in, n_c_out, 4, stride=1, bias=True)
        self.conv_gate = nn.Conv1d(n_c_in, n_c_out, 4, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool1d(2)

    def forward(self, input):
        padded_input = self.pad(input)
        x = self.conv_data(padded_input)
        g = self.conv_gate(padded_input)
        x = self.sigmoid(g) * x
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_c_in': 4, 'n_c_out': 4}]

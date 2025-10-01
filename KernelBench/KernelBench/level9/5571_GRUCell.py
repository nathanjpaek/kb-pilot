import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fc_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hn = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):
        x = x.view(-1, x.shape[1])
        i_r = self.fc_ir(x)
        h_r = self.fc_hr(h)
        i_z = self.fc_iz(x)
        h_z = self.fc_hz(h)
        i_n = self.fc_in(x)
        h_n = self.fc_hn(h)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_z + h_z)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (h - newgate)
        return hy


def get_inputs():
    return [torch.rand([4, 4, 64, 4]), torch.rand([4, 4, 1024, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]

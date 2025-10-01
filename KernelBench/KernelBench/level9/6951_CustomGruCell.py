import torch
import numpy as np
import torch.nn as nn


class CustomGruCell(nn.Module):
    """
    A forward only GRU cell.
    Input should be: (sequence length x batch size x input_size).
    The output is the output of the final forward call.
    It's not clear if it would be possible to use the output from each cell in a Plan
    because of the assumptions of 2D tensors in backprop.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomGruCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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
        i_r = self.fc_ir(x)
        h_r = self.fc_hr(h)
        i_z = self.fc_iz(x)
        h_z = self.fc_hz(h)
        i_n = self.fc_in(x)
        h_n = self.fc_hn(h)
        resetgate = (i_r + h_r).sigmoid()
        inputgate = (i_z + h_z).sigmoid()
        newgate = (i_n + resetgate * h_n).tanh()
        hy = newgate + inputgate * (h - newgate)
        return hy


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]

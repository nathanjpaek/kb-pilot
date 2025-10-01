import torch
import torch.utils.data
import torch.nn
import torch.optim
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, cell_size=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(cell_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        f_gate = self.sigmoid(self.gate(combined))
        i_gate = self.sigmoid(self.gate(combined))
        o_gate = self.sigmoid(self.gate(combined))
        cell_sub = self.tanh(self.gate(combined))
        cell = torch.add(torch.mul(cell, f_gate), torch.mul(cell_sub, i_gate))
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.sigmoid(self.output(hidden))
        return output, hidden, cell

    def initHidden(self, dim_num):
        return Variable(torch.zeros(dim_num, self.hidden_size))

    def initCell(self, dim_num):
        return Variable(torch.zeros(dim_num, self.cell_size))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 2])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]

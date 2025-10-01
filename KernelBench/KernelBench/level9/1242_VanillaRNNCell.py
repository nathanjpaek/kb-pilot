import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, nonlinearity='tanh', ct=False):
        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.ct = ct
        self.weight_ih = nn.Parameter(torch.zeros((input_size, hidden_size)))
        self.weight_hh = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()
        if self.nonlinearity == 'tanh':
            self.act = F.tanh
        elif self.nonlinearity == 'relu':
            self.act = F.relu
        else:
            raise RuntimeError('Unknown nonlinearity: {}'.format(self.
                nonlinearity))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inp, hidden_in):
        if not self.ct:
            hidden_out = self.act(torch.matmul(inp, self.weight_ih) + torch
                .matmul(hidden_in, self.weight_hh) + self.bias)
        else:
            alpha = 0.1
            hidden_out = (1 - alpha) * hidden_in + alpha * self.act(torch.
                matmul(inp, self.weight_ih) + torch.matmul(hidden_in, self.
                weight_hh) + self.bias)
        return hidden_out

    def init_hidden(self, batch_s):
        return torch.zeros(batch_s, self.hidden_size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True
        ):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(self.input_channels + self.hidden_channels, 
            4 * self.hidden_channels, self.kernel_size, 1, self.padding,
            bias=True)

    def forward(self, x, h, c):
        stacked_inputs = torch.cat((x, h), 1)
        gates = self.Gates(stacked_inputs)
        xi, xf, xo, xg = gates.chunk(4, 1)
        xi = torch.sigmoid(xi)
        xf = torch.sigmoid(xf)
        xo = torch.sigmoid(xo)
        xg = torch.tanh(xg)
        c = xf * c + xi * xg
        h = xo * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, hidden, shape):
        return Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])
            ), Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 3, 3])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'hidden_channels': 4, 'kernel_size': 4}]

import torch
import torch.nn as nn
from torch.autograd import Variable


class CLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True
        ):
        super(CLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 
            self.num_features * self.hidden_channels, self.kernel_size, 1,
            self.padding)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)
        Ai, Af, Ao, Ag = torch.split(A, A.size()[1] // self.num_features, dim=1
            )
        i = torch.sigmoid(Ai)
        f = torch.sigmoid(Af)
        o = torch.sigmoid(Ao)
        g = torch.tanh(Ag)
        c = c * f + i * g
        h = o * torch.tanh(c)
        return h, c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return Variable(torch.zeros(batch_size, hidden_c, shape[0],
                shape[1])), Variable(torch.zeros(batch_size, hidden_c,
                shape[0], shape[1]))
        except:
            return Variable(torch.zeros(batch_size, hidden_c, shape[0],
                shape[1])), Variable(torch.zeros(batch_size, hidden_c,
                shape[0], shape[1]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 3, 3])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'hidden_channels': 4, 'kernel_size': 4}]

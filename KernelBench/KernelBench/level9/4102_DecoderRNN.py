import torch
from torch import nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):

    def __init__(self, T, d):
        super().__init__()
        self.T = T
        self.d = d
        self.W = nn.Linear(d, d)
        self.U = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.b = nn.Parameter(torch.rand(1, d))

    def forward(self, x, h_t):
        y_t = None
        for t in range(self.T):
            a_t = self.b + self.W(h_t) + self.U(x[t])
            h_t = torch.tanh(a_t)
            o_t = self.V(h_t)
            y_t = F.softmax(o_t, 1)
        return y_t, h_t

    def init_hidden(self):
        return torch.zeros(1, self.d)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4, 'd': 4}]

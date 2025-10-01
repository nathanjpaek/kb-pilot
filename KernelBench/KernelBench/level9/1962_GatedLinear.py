import torch
from torch import nn
from torch.nn import init as init


class GatedLinear(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, out_ch)
        self.lin2 = nn.Linear(in_ch, out_ch)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.lin1(x)) * self.sig(self.lin2(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]

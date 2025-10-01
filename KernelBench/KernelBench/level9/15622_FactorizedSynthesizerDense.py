import torch
import torch.nn as nn


class FactorizedSynthesizerDense(nn.Module):

    def __init__(self, in_dims, sentence_length):
        super(FactorizedSynthesizerDense, self).__init__()
        self.a = 4
        self.b = sentence_length // self.a
        self.a_proj = nn.Linear(in_dims, self.a)
        self.b_proj = nn.Linear(in_dims, self.b)
        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        A = self.a_proj(x).repeat([1, 1, self.b])
        B = self.b_proj(x).repeat([1, 1, self.a])
        energy = A * B
        attention = self.softmax(energy)
        value = self.value_fc(x)
        out = torch.bmm(attention, value)
        return out, attention


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dims': 4, 'sentence_length': 4}]

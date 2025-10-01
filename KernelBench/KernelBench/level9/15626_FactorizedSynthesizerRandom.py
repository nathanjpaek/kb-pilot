import torch
import torch.nn as nn


class FactorizedSynthesizerRandom(nn.Module):

    def __init__(self, in_dims):
        super(FactorizedSynthesizerRandom, self).__init__()
        self.k = 8
        self.query_fc = nn.Linear(in_dims, self.k)
        self.key_fc = nn.Linear(in_dims, self.k)
        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_fc(x)
        out = torch.bmm(attention, value)
        return out, attention


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dims': 4}]

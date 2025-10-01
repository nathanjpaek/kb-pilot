import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, in_dims):
        super(Transformer, self).__init__()
        self.temperature = in_dims ** 0.5
        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)
        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x).permute(0, 2, 1)
        energy = torch.bmm(query / self.temperature, key)
        attention = self.softmax(energy)
        value = self.value_fc(x)
        out = torch.bmm(attention, value)
        return out, attention


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dims': 4}]

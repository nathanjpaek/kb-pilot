import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt


class Attn(nn.Module):

    def __init__(self, hidden_size, batch_first=True):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        if self.batch_first:
            batch_size, _seq_size = x.size()[:2]
        else:
            _seq_size, batch_size = x.size()[:2]
        weights = torch.bmm(x, self.weights.unsqueeze(0).repeat(batch_size,
            1, 1))
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        weighted_input = torch.mul(x, attentions.unsqueeze(-1).expand_as(x))
        return weighted_input, attentions


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

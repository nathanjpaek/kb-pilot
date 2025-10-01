import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import *
from math import *


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, context):
        """
        hidden: [batch, hidden_size]
        context: [seq, batch, hidden_size]

        return the context vector for decoding: [batch, hidden]
        """
        timestep = context.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        context = context.transpose(0, 1)
        attn_energies = self.score(h, context)
        score = F.softmax(attn_energies, dim=1).unsqueeze(1)
        context = torch.bmm(score, context).squeeze(1)
        return context

    def score(self, hidden, context):
        """
        hidden: [batch, seq, hidden]
        context: [batch, seq, hidden]
        """
        energy = torch.tanh(self.attn(torch.cat([hidden, context], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(context.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

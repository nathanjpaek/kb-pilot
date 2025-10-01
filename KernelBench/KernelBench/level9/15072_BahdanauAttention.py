import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import *


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size * 2))
        stdv = 1.0 / math.sqrt(hidden_size)
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.expand(src_len, -1, -1)
        hidden_energy = self.w1(hidden)
        encoder_outputs_energy = self.w2(encoder_outputs)
        energy = torch.cat((hidden_energy, encoder_outputs_energy), dim=2)
        v = self.v * energy
        v = torch.sum(v, dim=2)
        attention_energies = F.softmax(v, dim=0)
        return attention_energies


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

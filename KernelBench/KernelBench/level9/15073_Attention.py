import torch
import torch.nn as nn
from random import *


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, hidden, mask=None):
        encoder_energy = self.w1(encoder_outputs)
        decoder_energy = self.w2(hidden.squeeze(1))
        decoder_energy = decoder_energy.unsqueeze(1)
        combined = torch.tanh(encoder_energy + decoder_energy)
        energy = self.v(combined)
        energy = energy.squeeze(-1)
        if mask is not None:
            energy = energy.masked_fill(mask, -10000000000.0)
        attention = torch.softmax(energy, dim=-1)
        return attention


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

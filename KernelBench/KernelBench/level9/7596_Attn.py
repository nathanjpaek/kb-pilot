import torch
from torch import nn
import torch.nn.functional as F


class Attn(nn.Module):

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden, encoder_outputs, normalize=True):
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(hidden, encoder_outputs)
        normalized_energy = F.softmax(attn_energies, dim=2)
        context = torch.bmm(normalized_energy, encoder_outputs)
        return context.transpose(0, 1)

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = self.attn(torch.cat([H, encoder_outputs], 2))
        energy = self.v(F.tanh(energy)).transpose(1, 2)
        return energy


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

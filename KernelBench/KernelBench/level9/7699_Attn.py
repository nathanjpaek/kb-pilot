import torch
import torch.nn.functional as F
from torch import nn


class Attn(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        :param hidden: tensor of size [n_layer, B, H]
        :param encoder_outputs: tensor of size [B,T, H]
        """
        attn_energies = self.score(hidden, encoder_outputs)
        if mask is None:
            normalized_energy = F.softmax(attn_energies, dim=2)
        else:
            attn_energies.masked_fill_(mask, -1e+20)
            normalized_energy = F.softmax(attn_energies, dim=2)
        context = torch.bmm(normalized_energy, encoder_outputs)
        return context

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))
        energy = self.v(energy).transpose(1, 2)
        return energy


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

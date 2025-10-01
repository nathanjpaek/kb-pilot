import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_output):
        attn_energies = torch.sum(hidden * encoder_output, dim=2)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

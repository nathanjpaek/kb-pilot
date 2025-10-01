import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):

    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size

    def dot_product_attention(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def forward(self, hidden, encoded_output):
        energies = self.dot_product_attention(hidden, encoded_output)
        energies = energies.t()
        return F.softmax(energies, dim=1).unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    """ The score function for the attention mechanism.

    We define the score function as the general function from Luong et al.
    Where score(s_{i}, h_{j}) = s_{i} * W * h_{j}

    """

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        _batch_size, seq_len, _hidden_size = encoder_outputs.size()
        hidden = hidden.unsqueeze(1)
        hiddens = hidden.repeat(1, seq_len, 1)
        attn_energies = self.score(hiddens, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        hidden = hidden.unsqueeze(2)
        energy = energy.unsqueeze(3)
        energy = torch.matmul(hidden, energy)
        return energy.squeeze(3).squeeze(2)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

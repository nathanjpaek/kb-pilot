import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Generate positional encoding for a vector
    Args:
        length (int): length of the input sentence to be encoded
        d_model (int): dimention of the word vector
    Returns:
        torch.Tensor: positionaly encoded vector
    """

    def __init__(self, length, hidden_size):
        super(PositionalEncoder, self).__init__()
        f = torch.Tensor([(10000 ** (-i / hidden_size) if i % 2 == 0 else -
            10000 ** ((1 - i) / hidden_size)) for i in range(hidden_size)]
            ).unsqueeze(dim=1)
        phase = torch.Tensor([(0 if i % 2 == 0 else math.pi / 2) for i in
            range(hidden_size)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(hidden_size, 1)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos,
            f), phase)), requires_grad=False)

    def forward(self, x):
        return x + self.pos_encoding[0:x.size(1)]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'length': 4, 'hidden_size': 4}]

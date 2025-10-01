import torch
from torch import nn
import torch.utils.data


class ln(nn.Module):
    """
    Layer Normalization
    """

    def __init__(self, input):
        super(ln, self).__init__()
        self.ln = nn.LayerNorm(input.size()[1:])

    def forward(self, x):
        x = self.ln(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input': torch.rand([4, 4])}]

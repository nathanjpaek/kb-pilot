import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearGLUBlock(nn.Module):
    """A linear GLU block.

    Args:
        size (int): input and output dimension

    """

    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size * 2)

    def forward(self, xs):
        return F.glu(self.fc(xs), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]

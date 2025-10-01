import torch
from torch import nn


class TestNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        x_len = x.shape[-1]
        return self.conv(x.view(-1, 1, x_len)).view(x.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

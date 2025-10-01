import torch
import torch.nn as nn


class Channel_mean(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def forward(self, V):
        """
        only V[0]
        """
        return torch.sum(V[0], dim=0).squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

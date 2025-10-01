import torch
from torch import nn


class AppendDim(nn.Module):
    """
    Append a new dim to states with size out_dim
    """

    def __init__(self, out_dim=1):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, states, **kwargs):
        x = states.unsqueeze(len(states.size()))
        x = x.repeat(*([1] * len(states.size()) + [self.out_dim]))
        return x

    def reset_parameters(self, *args, **kwargs):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

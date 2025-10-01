import torch
from torch import nn


class AvgReducePool1d(nn.Module):
    """A subclass of :torch_nn:`Module`.
    Avg Pool layer for 1D inputs. The same as :torch_nn:`AvgPool1d` except that
    the pooling dimension is entirely reduced (i.e., `pool_size=input_length`).
    """

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return torch.mean(input, dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

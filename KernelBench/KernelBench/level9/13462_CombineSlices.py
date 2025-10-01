import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import torch.optim
import torch.fft


class CombineSlices(nn.Module):

    def __init__(self, slice_dim=2):
        super().__init__()
        self.slice_dim = slice_dim

    def forward(self, x):
        return torch.index_select(x, dim=self.slice_dim, index=torch.tensor
            (0, device=x.device))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

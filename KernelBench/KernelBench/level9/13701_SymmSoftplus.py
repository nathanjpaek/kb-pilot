import torch
from torch.utils.data import Dataset as Dataset
import torch.utils.data


def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


class SymmSoftplus(torch.nn.Module):

    def forward(self, x):
        return symm_softplus(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn
import torch.nn as nn
from itertools import repeat


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)
    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)


class BondEnergyModule(nn.Module):

    def __init__(self, batch=True):
        super().__init__()

    def forward(self, xyz, bond_adj, bond_len, bond_par):
        e = (xyz[bond_adj[:, 0]] - xyz[bond_adj[:, 1]]).pow(2).sum(1).sqrt()[
            :, None]
        ebond = bond_par * (e - bond_len) ** 2
        energy = 0.5 * scatter_add(src=ebond, index=bond_adj[:, 0], dim=0,
            dim_size=xyz.shape[0])
        energy += 0.5 * scatter_add(src=ebond, index=bond_adj[:, 1], dim=0,
            dim_size=xyz.shape[0])
        return energy


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype
        =torch.int64), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]

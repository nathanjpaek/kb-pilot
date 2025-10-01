import torch
from torch import nn
from typing import *


class Grouping(nn.Module):

    def __init__(self, n_groups):
        super().__init__()
        self.n_groups = n_groups

    def forward(self, x):
        x = x.permute(2, 0, 1)
        n_modalities = len(x)
        out = []
        for i in range(self.n_groups):
            start_modality = n_modalities * i // self.n_groups
            end_modality = n_modalities * (i + 1) // self.n_groups
            sel = list(x[start_modality:end_modality])
            sel = torch.stack(sel, dim=len(sel[0].size()))
            out.append(sel)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_groups': 1}]

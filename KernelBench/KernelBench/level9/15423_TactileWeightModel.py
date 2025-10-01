import torch
import torch.utils.data
import torch.nn as nn
from typing import Optional
import torch.linalg


class TactileWeightModel(nn.Module):

    def __init__(self, device: 'torch.device', dim: 'int'=3, wt_init:
        'Optional[torch.Tensor]'=None):
        super().__init__()
        wt_init_ = torch.rand(1, dim)
        if wt_init is not None:
            wt_init_ = wt_init
        self.param = nn.Parameter(wt_init_)
        self

    def forward(self):
        return self.param.clone()


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'device': 0}]

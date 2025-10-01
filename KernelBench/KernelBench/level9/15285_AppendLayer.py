import torch
import numpy as np
import torch.nn as nn


class AppendLayer(nn.Module):

    def __init__(self, noise=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var = nn.Parameter(torch.DoubleTensor(1, 1))
        nn.init.constant_(self.log_var, val=np.log(noise))

    def forward(self, x):
        return torch.cat((x, self.log_var * torch.ones_like(x)), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

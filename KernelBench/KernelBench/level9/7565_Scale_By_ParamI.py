import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class Scale_By_ParamI(nn.Module):

    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = x * self.scalar
        return out

    def ibp_forward(self, l, u):
        if self.scalar >= 0:
            l_ = l * self.scalar
            u_ = u * self.scalar
        else:
            u_ = l * self.scalar
            l_ = u * self.scalar
        return l_, u_


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

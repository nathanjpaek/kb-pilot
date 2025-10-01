import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class Add_ParamI(nn.Module):

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = x + self.bias
        return out

    def ibp_forward(self, l, u):
        l_ = l + self.bias
        u_ = u + self.bias
        return l_, u_


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

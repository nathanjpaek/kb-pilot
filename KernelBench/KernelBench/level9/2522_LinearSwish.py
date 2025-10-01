import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
import torch.backends.cudnn
import torch.backends.mkl


class LinearSwish(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearSwish, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        linear_res = self.linear(x)
        return F.silu(linear_res)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]

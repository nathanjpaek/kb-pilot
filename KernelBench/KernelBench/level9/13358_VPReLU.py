import torch
import torch.nn as nn
import torch.nn.functional as F


class VPReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: 'bool'

    def __init__(self, inplace: 'bool'=False):
        super(VPReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.relu(input, inplace=self.inplace) * 1.7139588594436646

    def extra_repr(self) ->str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

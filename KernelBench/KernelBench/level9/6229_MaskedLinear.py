import torch
import torch.cuda
from torch.nn.functional import *


class MaskedLinear(torch.nn.Linear):

    def forward(self, x, mask):
        out = super().forward(x)
        if mask.is_floating_point():
            out = out * mask
        else:
            out = out * mask.type_as(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]

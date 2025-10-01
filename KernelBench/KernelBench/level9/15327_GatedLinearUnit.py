import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):

    def forward(self, x, mask):
        x = th.cat((x, mask), 1)
        return F.glu(x, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class LN(nn.Module):

    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

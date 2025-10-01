import torch
import torch.nn.functional as F
import torch.nn as nn


class SoftGeneratorAttention(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1])
        return bn_module(x)

    def forward(self, key, x):
        attn = torch.mul(key, x).sum(dim=1)
        attn = F.softmax(attn)
        return attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

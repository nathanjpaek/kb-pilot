import torch
from torch import nn
import torch as t


class ImgPatchConverter(nn.Module):

    def __init__(self):
        super(ImgPatchConverter, self).__init__()

    def forward(self, x):
        x = t.flatten(x, start_dim=2)
        x = t.transpose(x, 1, 2).contiguous()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

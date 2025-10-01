import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMaxLayer(nn.Module):

    def forward(self, tensor, dim=1):
        return F.softmax(tensor, dim=dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

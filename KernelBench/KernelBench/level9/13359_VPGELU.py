import torch
import torch.nn as nn
import torch.nn.functional as F


class VPGELU(nn.Module):

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.gelu(input) * 1.7015043497085571


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

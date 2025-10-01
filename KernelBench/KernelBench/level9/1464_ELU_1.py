import torch
import torch.nn.functional as F
import torch.nn as nn


class ELU_1(nn.ELU):

    def __init__(self, *args, **kwargs):
        super(ELU_1, self).__init__(*args, **kwargs)

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

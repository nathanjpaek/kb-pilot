import torch
import torch.utils.data
import torch.nn as nn


class gconv_RNN(nn.Module):

    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]

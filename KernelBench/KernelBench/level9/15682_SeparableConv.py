import torch
from torch import nn


class SeparableConv(nn.Module):

    def __init__(self, nb_dim, nb_out, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(nb_dim, nb_dim, kernel_size, groups=nb_dim,
            padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv1d(nb_dim, nb_out, 1, groups=1, padding=0, bias
            =True)

    def forward(self, x):
        """
        :param x: shape(bsz, seqlen, 500)
        """
        out = self.conv1(x.transpose(1, 2))
        out = self.conv2(out)
        return out.transpose(1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'nb_dim': 4, 'nb_out': 4, 'kernel_size': 4}]

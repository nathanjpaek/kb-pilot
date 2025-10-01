import torch
import torch.nn as nn


class l2_norm_layer(nn.Module):

    def __init__(self):
        super(l2_norm_layer, self).__init__()

    def forward(self, x):
        """
        :param x:  B x D
        :return:
        """
        norm_x = torch.sqrt((x ** 2).sum(1) + 1e-10)
        return x / norm_x[:, None]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

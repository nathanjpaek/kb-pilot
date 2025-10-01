import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class avgpool(nn.Module):
    """
    Mean pooling class - downsampling
    """

    def __init__(self, up_size=0):
        super(avgpool, self).__init__()

    def forward(self, x):
        out_man = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1:
            :2] + x[:, :, 1::2, 1::2]) / 4
        return out_man


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn


class custom_loss(nn.Module):

    def __init__(self):
        super(custom_loss, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels do not divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

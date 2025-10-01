import torch
import torch.nn as nn


class GlobalMaxPooling(nn.Module):

    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, x):
        res, _ = torch.max(x, dim=1)
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

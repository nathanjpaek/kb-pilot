import torch
import torch.nn as nn


class MinibatchStd(nn.Module):
    """
    calculate minibatch std to avoid mode collapse
    """

    def __init__(self):
        super(MinibatchStd, self).__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1
        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

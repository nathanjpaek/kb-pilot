import torch
import torch.nn as nn


class MeanAggregator(nn.Module):

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, x: 'torch.Tensor'):
        return x.mean(dim=1)

    def __call__(self, *args, **kwargs):
        return super(MeanAggregator, self).__call__(*args, **kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

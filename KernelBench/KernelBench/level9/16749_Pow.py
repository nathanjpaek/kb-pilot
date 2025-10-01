import torch
import torch.nn as nn


class Pow(nn.Module):
    """
    Applies `x ** sigmoid(a)`, with `a` fixed or trainable.
    """

    def __init__(self, a=0, trainable=False):
        super(Pow, self).__init__()
        if trainable:
            a = nn.Parameter(torch.tensor(a, dtype=torch.get_default_dtype()))
        self.a = a
        self.trainable = trainable

    def forward(self, x):
        if self.trainable or self.a != 0:
            x = torch.pow(x, torch.sigmoid(self.a))
        else:
            x = torch.sqrt(x)
        return x

    def extra_repr(self):
        return 'trainable={}'.format(repr(self.trainable))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

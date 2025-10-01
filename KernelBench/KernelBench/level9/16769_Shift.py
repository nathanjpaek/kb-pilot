import torch
import torch.nn as nn


class Shift(nn.Module):

    def __init__(self, amount, inplace=False):
        super(Shift, self).__init__()
        self.amount = amount
        self.inplace = inplace

    def extra_repr(self):
        return 'amount={}'.format(self.amount)

    def forward(self, x):
        if self.inplace:
            x += self.amount
        else:
            x = x + self.amount
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'amount': 4}]

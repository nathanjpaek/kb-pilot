import torch
import torch.nn as nn


class BCELoss(nn.BCELoss):

    def __init__(self, **kwargs):
        super(BCELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.float()
        return super(BCELoss, self).forward(input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

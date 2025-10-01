import torch
import torch.nn as nn


class DummyLoss(nn.Module):
    """
    Dummy Loss for debugging
    """

    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, inp, target):
        delta = inp - target
        None
        return delta.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

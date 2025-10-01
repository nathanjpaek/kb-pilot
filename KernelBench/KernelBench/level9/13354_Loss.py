import torch
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self, lambd):
        super(Loss, self).__init__()
        self.lambd = lambd
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, O, Y, C):
        return (Y * (self.lambd * C - self.lsm(O))).mean(dim=0).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lambd': 4}]

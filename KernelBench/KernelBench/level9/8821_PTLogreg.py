import torch
import torch.nn as nn


class PTLogreg(nn.Module):

    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super(PTLogreg, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(D, C))
        self.b = torch.nn.Parameter(torch.zeros(C))

    def forward(self, X):
        return nn.functional.softmax(torch.mm(X, self.W) + self.b)

    def get_loss(self, X, Yoh_):
        Y = self.forward(X)
        return -torch.mean(torch.sum(Yoh_ * torch.log(Y) + (1 - Yoh_) *
            torch.log(1 - Y)))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'D': 4, 'C': 4}]

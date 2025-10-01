import torch
import torch.nn as nn


class MultinomialCELoss(nn.Module):

    def __init__(self):
        super(MultinomialCELoss, self).__init__()

    def forward(self, x, y):
        x = x + 1e-08
        x = torch.log(x)
        zlogz = y * x
        loss = -zlogz.sum()
        loss /= x.shape[0] * x.shape[2] * x.shape[3]
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

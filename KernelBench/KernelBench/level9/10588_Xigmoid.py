import torch
import torch.nn as nn


def xigmoid(x, alpha=1.0):
    cond = x > 0
    ax = alpha * x
    if_x = torch.exp(ax)
    else_x = 1.0 / if_x
    if_x = if_x - 1.0
    else_x = 1.0 - else_x
    cond_x = torch.where(cond, if_x, else_x)
    return torch.sigmoid(alpha * cond_x)


class Xigmoid(nn.Module):

    def __init__(self, alpha=1.0):
        super(Xigmoid, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return xigmoid(x, self.alpha)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

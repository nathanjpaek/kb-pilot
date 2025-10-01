import torch
import torch.nn as nn


class L_2nd(nn.Module):

    def __init__(self, beta):
        super(L_2nd, self).__init__()
        self.beta = beta

    def forward(self, y_pred, y_true):
        b = torch.ones_like(y_true)
        b[y_true != 0] = self.beta
        x = ((y_true - y_pred) * b) ** 2
        t = torch.sum(x, dim=-1)
        return torch.mean(t)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'beta': 4}]

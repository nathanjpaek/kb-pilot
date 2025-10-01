import torch
import torch.nn as nn


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x_true, x_pred):
        return torch.sqrt(torch.mean(torch.pow(x_pred - x_true, 2), dim=-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

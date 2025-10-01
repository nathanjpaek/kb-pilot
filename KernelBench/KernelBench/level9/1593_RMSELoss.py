import torch
import torch.nn as nn


class RMSELoss(nn.Module):

    def __init__(self, eps=1e-08):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y) + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


class RMSLELoss(nn.Module):

    def __init__(self, eps=1e-08):
        super(RMSLELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(torch.log(y_hat + 1), torch.log(y + 1)) +
            self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

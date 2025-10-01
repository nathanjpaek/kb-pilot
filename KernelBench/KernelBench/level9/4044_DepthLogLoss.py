import torch
import torch.nn as nn


class DepthLogLoss(nn.Module):

    def __init__(self, balance_factor):
        super(DepthLogLoss, self).__init__()
        self.balance_factor = balance_factor

    def forward(self, inputs, targets):
        n, _, h, w = inputs.shape
        n_pixel = n * h * w
        inputs = torch.log(inputs + 1e-08)
        targets = torch.log(targets)
        d = inputs - targets
        loss = torch.sum(d ** 2) / n_pixel - self.balance_factor * torch.sum(d
            ) ** 2 / n_pixel ** 2
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'balance_factor': 4}]

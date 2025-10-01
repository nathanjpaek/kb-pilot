import torch
import torch.nn as nn


class selfVLoss(nn.Module):

    def __init__(self, lambda_v, lambda_r):
        super(selfVLoss, self).__init__()
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r

    def forward(self, v, z):
        return 1.0 * self.lambda_v / self.lambda_r * torch.mean(torch.sum(
            torch.pow(v - z, 2), 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lambda_v': 4, 'lambda_r': 4}]

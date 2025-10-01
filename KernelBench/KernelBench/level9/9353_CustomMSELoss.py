import torch
import torch.nn as nn


class CustomMSELoss(nn.Module):

    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow(torch.log(torch.exp(x) - torch.exp(y)), 2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


class ATANLoss(nn.Module):

    def __init__(self):
        super(ATANLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = torch.mean(torch.atan(torch.abs(inputs - targets)))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

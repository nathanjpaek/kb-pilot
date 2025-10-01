import torch
import torch.nn as nn


class EPELoss(nn.Module):

    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target + 1e-16, p=2, dim=1).mean()
        return lossvalue


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


class DotProductLoss(nn.Module):

    def __init__(self):
        super(DotProductLoss, self).__init__()

    def forward(self, output, target):
        return -torch.dot(target.view(-1), output.view(-1)) / target.nelement()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

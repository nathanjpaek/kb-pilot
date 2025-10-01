import math
import torch
import torch.nn as nn


def calc_angular_difference(a1, a2):
    distance = torch.min(torch.abs(a1 - a2), torch.tensor(2 * math.pi) -
        torch.abs(a2 - a1))
    diff = torch.sqrt(torch.abs(distance * distance))
    return diff


class AngularLoss(nn.Module):

    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, predicted, actual, mask=None):
        return calc_angular_difference(predicted, actual)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

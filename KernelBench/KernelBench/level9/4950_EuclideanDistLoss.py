import torch
from torch import nn


class EuclideanDistLoss(nn.Module):

    def __init__(self):
        super(EuclideanDistLoss, self).__init__()

    def forward(self, inputs, inputs_rot):
        dist = torch.dist(inputs, inputs_rot, p=2.0)
        return dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

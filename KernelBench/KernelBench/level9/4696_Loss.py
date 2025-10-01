import torch
import torch.nn as nn
from torch.nn import functional as F


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, label):
        loss = F.cross_entropy(output, label)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

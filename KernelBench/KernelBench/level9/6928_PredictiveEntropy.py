import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class PredictiveEntropy(nn.Module):

    def __init__(self):
        super(PredictiveEntropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

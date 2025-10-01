import torch
import torch.nn as nn


class NegSamplingLoss(nn.Module):

    def __init__(self):
        super(NegSamplingLoss, self).__init__()

    def forward(self, score, sign):
        return -torch.mean(torch.sigmoid(sign * score))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

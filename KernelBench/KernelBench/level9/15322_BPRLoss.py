import torch
import torch.utils.data
import torch
import torch.nn as nn


class BPRLoss(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.m = nn.LogSigmoid()

    def forward(self, positives, negatives):
        return -self.m(positives - negatives).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
from torch.nn import functional as F
from torch import nn
from typing import *


class GumbelSoftMaxSampler(nn.Module):

    def __init__(self, hard=False):
        super().__init__()
        self.hard = hard

    def forward(self, logits):
        return F.gumbel_softmax(logits=logits, hard=self.hard)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReluWithStats(nn.Module):

    def __init__(self):
        super(ReluWithStats, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []

    def forward(self, preact):
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())
        act = F.relu(preact)
        return act


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

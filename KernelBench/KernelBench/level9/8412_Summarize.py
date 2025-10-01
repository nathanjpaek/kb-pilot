import torch
import torch.nn as nn


class Summarize(nn.Module):

    def __init__(self):
        super(Summarize, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, vec):
        return self.sigmoid(torch.mean(vec, dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

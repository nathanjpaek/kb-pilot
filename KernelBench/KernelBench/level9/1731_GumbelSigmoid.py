import torch
import torch.nn as nn


def gumbel(x, eps=1e-20):
    return -torch.log(-torch.log(torch.rand_like(x) + eps) + eps)


class GumbelSigmoid(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, logits):
        y = logits + gumbel(logits) if self.training else logits
        return torch.sigmoid(self.scale * y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Entropy_loss(nn.Module):

    def __init__(self):
        super(Entropy_loss, self).__init__()

    def forward(self, x):
        probs = F.softmax(x, dim=1)
        b = torch.log(probs) * probs
        b = -1.0 * b.sum(dim=1)
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

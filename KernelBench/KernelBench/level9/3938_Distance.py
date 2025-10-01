import torch
import torch.nn as nn


class Distance(nn.Module):

    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, s, t):
        n, q = s.shape[0], t.shape[0]
        dist = (t.unsqueeze(0).expand(n, q, -1) - s.unsqueeze(1).expand(n,
            q, -1)).pow(2).sum(dim=2).T
        return dist


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]

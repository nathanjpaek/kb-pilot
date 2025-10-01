import torch
import torch.nn as nn
import torch.jit
import torch.nn


def normalized_distance(data, distance):
    data = data.view(data.size(0), -1)
    reference = data[:, None]
    comparison = data[None, :]
    result = distance(reference, comparison)
    result = result / result.sum(dim=1, keepdim=True).detach()
    return result


class NormalizedDistance(nn.Module):

    def __init__(self, distance=None):
        super().__init__()
        self.distance = distance
        if self.distance is None:
            self.distance = lambda x, y: (x - y).norm(dim=-1)

    def forward(self, data):
        return normalized_distance(data, self.distance)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

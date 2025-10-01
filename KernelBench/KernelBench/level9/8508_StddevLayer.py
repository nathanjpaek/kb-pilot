import torch
from torch import nn
import torch.nn


class StddevLayer(nn.Module):

    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = 4
        self.num_new_features = 1

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features, c // self.
            num_new_features, h, w])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-08) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, self.
            num_new_features, h, w)
        z = torch.cat([x, y], dim=1)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

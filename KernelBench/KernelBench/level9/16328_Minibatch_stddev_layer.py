import torch
import torch.nn as nn


class Minibatch_stddev_layer(nn.Module):
    """
        Minibatch standard deviation layer. (D_stylegan2)
    """

    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        y = x.view(group_size, -1, self.num_new_features, c // self.
            num_new_features, h, w)
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + 1e-08)
        y = torch.mean(y, dim=[2, 3, 4], keepdim=True)
        y = torch.mean(y, dim=2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

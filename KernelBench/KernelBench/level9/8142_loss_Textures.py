import torch
import torch.utils.data
import torch.nn as nn
import torch.nn


class loss_Textures(nn.Module):

    def __init__(self, nc=1, alpha=1.2, margin=0):
        super(loss_Textures, self).__init__()
        self.nc = nc
        self.alpha = alpha
        self.margin = margin

    def forward(self, x, y):
        xi = x.contiguous().view(x.size(0), -1, self.nc, x.size(2), x.size(3))
        yi = y.contiguous().view(y.size(0), -1, self.nc, y.size(2), y.size(3))
        xi2 = torch.sum(xi * xi, dim=2)
        yi2 = torch.sum(yi * yi, dim=2)
        out = nn.functional.relu(yi2.mul(self.alpha) - xi2 + self.margin)
        return torch.mean(out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

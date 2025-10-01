import torch
import torch.nn as nn


class KL_Divergence(nn.Module):

    def __init__(self, sum_dim=None, sqrt=False, dimension_warn=0):
        super().__init__()
        self.sum_dim = sum_dim
        self.sqrt = sqrt
        self.dimension_warn = dimension_warn

    def forward(self, x, y):
        x = x.view(x.size(0), x.size(1), -1)
        x = x / x.norm(1, dim=-1).unsqueeze(-1)
        y = y.view(y.size(0), y.size(1), -1)
        y = y / y.norm(1, dim=-1).unsqueeze(-1)
        loss = torch.sum(y * (y.log() - x.log()), dim=self.sum_dim)
        return loss.squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

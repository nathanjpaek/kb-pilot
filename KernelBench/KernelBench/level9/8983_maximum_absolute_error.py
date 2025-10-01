import torch
from torch import nn


class maximum_absolute_error(nn.Module):

    def forward(self, yhat, y):
        return torch.max(torch.abs(torch.sub(y, yhat)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

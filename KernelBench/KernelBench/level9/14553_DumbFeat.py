import torch
import torch.nn as nn
import torch.optim


class DumbFeat(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=dropout, inplace=False)
        else:
            self.dropout = None

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        assert x.dim() == 2
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dropout': 0.5}]

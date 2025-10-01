import torch
import torch.nn as nn


class LinearMaxPoolLinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 4, bias=False)
        self.lin1.weight = nn.Parameter(torch.eye(4, 4))
        self.pool1 = nn.MaxPool1d(4)
        self.lin2 = nn.Linear(1, 1, bias=False)
        self.lin2.weight = nn.Parameter(torch.ones(1, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.lin2(self.pool1(self.lin1(x))[:, 0, :])


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]

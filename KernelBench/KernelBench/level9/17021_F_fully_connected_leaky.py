import torch
from torch import nn


class F_fully_connected_leaky(nn.Module):
    """Fully connected tranformation, not reversible, but used below."""

    def __init__(self, size_in, size, internal_size=None, dropout=0.0,
        batch_norm=False, leaky_slope=0.01):
        super(F_fully_connected_leaky, self).__init__()
        if not internal_size:
            internal_size = 2 * size
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc2d = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)
        self.nl1 = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2 = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2b = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2d = nn.ReLU()
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(internal_size)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm1d(internal_size)
            self.bn2.weight.data.fill_(1)
            self.bn2b = nn.BatchNorm1d(internal_size)
            self.bn2b.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.fc1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.nl1(self.d1(out))
        out = self.fc2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.nl2(self.d2(out))
        out = self.fc2b(out)
        if self.batch_norm:
            out = self.bn2b(out)
        out = self.nl2b(self.d2b(out))
        out = self.fc2d(out)
        out = self.nl2d(out)
        out = self.fc3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size_in': 4, 'size': 4}]

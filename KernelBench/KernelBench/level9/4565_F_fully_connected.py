import torch
import torch.nn as nn
import torch.optim


class F_fully_connected(nn.Module):
    """Fully connected tranformation, not reversible, but used below."""

    def __init__(self, size_in, size, internal_size=None, dropout=0.0):
        super().__init__()
        if not internal_size:
            internal_size = 2 * size
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)
        self.nl1 = nn.ReLU()
        self.nl2 = nn.ReLU()
        self.nl2b = nn.ReLU()

    def forward(self, x):
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.nl2(self.d2(self.fc2(out)))
        out = self.nl2b(self.d2b(self.fc2b(out)))
        out = self.fc3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size_in': 4, 'size': 4}]

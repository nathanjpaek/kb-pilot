import torch
import torch.nn as nn
import torch.utils.data


class AE_3D_small(nn.Module):

    def __init__(self, n_features=4):
        super(AE_3D_small, self).__init__()
        self.en1 = nn.Linear(n_features, 3)
        self.de1 = nn.Linear(3, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en1(x)

    def decode(self, x):
        return self.de1(self.tanh(x))

    def forward(self, x):
        return self.decode(self.encode(x))

    def describe(self):
        return 'in-3-out'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

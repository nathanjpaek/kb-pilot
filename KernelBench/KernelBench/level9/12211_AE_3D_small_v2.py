import torch
import torch.nn as nn
import torch.utils.data


class AE_3D_small_v2(nn.Module):

    def __init__(self, n_features=4):
        super(AE_3D_small_v2, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 3)
        self.de1 = nn.Linear(3, 8)
        self.de2 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en2(self.tanh(self.en1(x)))

    def decode(self, x):
        return self.de2(self.tanh(self.de1(self.tanh(x))))

    def forward(self, x):
        return self.decode(self.encode(x))

    def describe(self):
        return 'in-8-3-8-out'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

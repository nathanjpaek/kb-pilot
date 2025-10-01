import torch
import torch.nn as nn


class _MLP_B(nn.Module):
    """MLP that only use age gender MMSE"""

    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP_B, self).__init__()
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 2)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X):
        out = self.do1(X)
        out = self.fc1(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'drop_rate': 0.5, 'fil_num': 4}]

import torch
import torch.nn as nn


class SVM(nn.Module):

    def __init__(self, hidden_size):
        super(SVM, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.linear1(x))
        return y.view(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

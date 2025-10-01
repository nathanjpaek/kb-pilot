import torch
from torch import nn as nn
from torch.nn import functional as F


class MnistMlp(nn.Module):

    def __init__(self, width, dropout_p):
        super().__init__()
        self.fc1 = nn.Linear(784, width)
        self.fc2 = nn.Linear(width, 10)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = torch.reshape(x, (-1, 784))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {'width': 4, 'dropout_p': 0.5}]

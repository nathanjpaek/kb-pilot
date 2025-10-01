import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        return self.linear2(F.elu(self.linear1(x.view(x.size(0), -1))))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'n_classes': 4}]

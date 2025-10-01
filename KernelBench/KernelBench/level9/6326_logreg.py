import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import weight_norm


class logreg(nn.Module):

    def __init__(self, input_size, classes):
        super(logreg, self).__init__()
        linear = nn.Linear(input_size, classes)
        self.logistic_reg = weight_norm(linear, name='weight')

    def forward(self, x):
        return self.logistic_reg(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'classes': 4}]

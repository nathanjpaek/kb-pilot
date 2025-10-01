import torch
import torch.nn as nn
import torch.utils.data


class LogisticRegressionBinaryClassifier(nn.Module):

    def __init__(self, input_size):
        super(LogisticRegressionBinaryClassifier, self).__init__()
        self.input_size = input_size
        self.mapping = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.mapping(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]

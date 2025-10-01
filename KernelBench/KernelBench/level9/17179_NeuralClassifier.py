import torch
import torch.nn as nn
import torch.utils.data


class NeuralClassifier(nn.Module):

    def __init__(self, input_size, n_classes):
        super(NeuralClassifier, self).__init__()
        self.input_size = input_size
        self.mapping1 = nn.Linear(input_size, input_size)
        self.mapping2 = nn.Linear(input_size, n_classes)
        self.f = torch.sigmoid

    def forward(self, x):
        x = self.f(self.mapping1(x))
        return self.f(self.mapping2(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'n_classes': 4}]

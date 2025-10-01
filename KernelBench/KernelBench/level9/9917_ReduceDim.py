import torch
import torch.nn as nn
import torch.nn.functional as F


class ReduceDim(nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dimension': 4, 'output_dimension': 4}]

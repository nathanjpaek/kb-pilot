import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class Regression(nn.Module):

    def __init__(self, input_size, output_size):
        super(Regression, self).__init__()
        self.layer1 = nn.Linear(input_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, output_size)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]

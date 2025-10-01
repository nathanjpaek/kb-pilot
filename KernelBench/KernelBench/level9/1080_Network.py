import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.functional import softmax


class Network(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, t):
        t = relu(self.fc1(t))
        t = relu(self.fc2(t))
        t = softmax(self.out(t), dim=0)
        return t


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]

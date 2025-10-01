import torch
import torch.nn as nn


class MultiModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(MultiModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, output_size)
        self.out = nn.Softmax()

    def forward(self, input_):
        out = self.layer1(input_)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.out(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]

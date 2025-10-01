import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import *
import torch.optim.lr_scheduler
import torch.quantization
import torch.onnx
import torch.testing


class DummyDenseWithRelu(nn.Module):

    def __init__(self, input_size, output_size, relu=None):
        super(DummyDenseWithRelu, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu = relu or nn.ReLU()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.relu(self.linear(x))


class DummyModelWithSharedSubmodule(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DummyModelWithSharedSubmodule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dense1 = DummyDenseWithRelu(input_size, hidden_size)
        self.dense2 = DummyDenseWithRelu(hidden_size, output_size, self.
            dense1.relu)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]

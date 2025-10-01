import torch
import torch.nn as nn


class ANN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, inpt):
        hidden = self.i2h(inpt)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]

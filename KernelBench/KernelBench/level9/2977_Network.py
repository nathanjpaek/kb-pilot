import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, number_of_inputs, number_of_outputs):
        super(Network, self).__init__()
        self.l1 = nn.Linear(number_of_inputs, number_of_inputs)
        self.l2 = nn.Linear(number_of_inputs, number_of_inputs)
        self.l3 = nn.Linear(number_of_inputs, number_of_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.l2(out1)
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'number_of_inputs': 4, 'number_of_outputs': 4}]

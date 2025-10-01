import torch
import torch.nn as nn


class NeuralModel(nn.Module):

    def __init__(self, input):
        super(NeuralModel, self).__init__()
        self.dense1 = nn.Linear(in_features=input, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=16)
        self.dense3 = nn.Linear(in_features=16, out_features=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out1 = self.dense1(x)
        out1 = self.relu(out1)
        out2 = self.dense2(out1)
        out2 = self.relu(out2)
        out3 = self.dense3(out2)
        result = self.tanh(out3)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input': 4}]

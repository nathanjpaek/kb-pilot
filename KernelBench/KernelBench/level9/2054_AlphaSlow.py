import torch
import torch.nn as nn


class AlphaSlow(nn.Module):

    def __init__(self, n_in, n_out):
        super(AlphaSlow, self).__init__()
        self.fc1 = nn.Linear(n_in, 320, bias=True)
        self.fc2 = nn.Linear(320, 160, bias=True)
        self.fc3 = nn.Linear(160, 80, bias=True)
        self.fc4 = nn.Linear(80, 80, bias=True)
        self.fc5 = nn.Linear(80, 40, bias=True)
        self.fc6 = nn.Linear(40, 40, bias=True)
        self.fc7 = nn.Linear(40, 20, bias=True)
        self.fc8 = nn.Linear(20, 20, bias=True)
        self.fc9 = nn.Linear(20, n_out, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()

    def forward(self, inputs):
        outputs = self.ReLU(self.fc1(inputs))
        outputs = self.ReLU(self.fc2(outputs))
        outputs = self.ReLU(self.fc3(outputs))
        outputs = self.ReLU(self.fc4(outputs))
        outputs = self.ReLU(self.fc5(outputs))
        outputs = self.ReLU(self.fc6(outputs))
        outputs = self.ReLU(self.fc7(outputs))
        outputs = self.ReLU(self.fc8(outputs))
        outputs = self.Tanh(self.fc9(outputs))
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]

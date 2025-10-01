import torch
import torch.nn as nn


class ScaleLayer(nn.Module):

    def __init__(self, init_value=0.001):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.scaler1 = ScaleLayer(init_value=torch.tensor(2.0))
        self.scaler2 = ScaleLayer(init_value=torch.tensor(2.0))
        self.scaler3 = ScaleLayer(init_value=torch.tensor(2.0))

    def forward(self, x):
        x = self.scaler1(x)
        x = self.scaler2(x)
        x = self.scaler3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

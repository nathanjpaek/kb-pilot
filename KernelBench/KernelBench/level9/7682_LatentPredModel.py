import torch
import torch.nn as nn


class LatentPredModel(torch.nn.Module):

    def __init__(self, in_channels):
        super(LatentPredModel, self).__init__()
        self.layer1 = nn.Linear(in_channels, 32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(64, 64)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()
        self.layer6 = nn.Linear(32, in_channels)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        x = self.relu5(self.layer5(x))
        x = self.layer6(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]

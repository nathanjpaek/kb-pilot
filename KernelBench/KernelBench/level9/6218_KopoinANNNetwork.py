import torch
import torch.nn as nn


class KopoinANNNetwork(nn.Module):

    def __init__(self, featShape):
        super(KopoinANNNetwork, self).__init__()
        self.featShape = featShape
        self.act = nn.Sigmoid()
        self.layer0 = nn.Linear(featShape, featShape // 2)
        self.layer1 = nn.Linear(featShape // 2, featShape // 2)
        self.layer2 = nn.Linear(featShape // 2, 2)

    def forward(self, x):
        x = self.layer0(x)
        x = self.act(x)
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'featShape': 4}]

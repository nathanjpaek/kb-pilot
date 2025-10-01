import torch
from torch import nn


class Maxout(nn.Module):

    def __init__(self, in_features, out_features):
        super(Maxout, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(x)
        return torch.max(output1, output2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]

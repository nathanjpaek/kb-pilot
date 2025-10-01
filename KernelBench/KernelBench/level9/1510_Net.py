import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, feature_num):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(feature_num, 500)
        self.layer_2 = nn.Linear(500, 20)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_num': 4}]

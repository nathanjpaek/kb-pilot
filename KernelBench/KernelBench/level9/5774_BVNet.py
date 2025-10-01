import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn


class BVNet(nn.Module):
    """
    Baseline REINFORCE - Value Calculating Network
    """

    def __init__(self, input_size):
        super(BVNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.dr1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.dr2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(input_size // 4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dr1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.dr2(x)
        x = torch.tanh(self.fc3(x)) * 1.5
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]

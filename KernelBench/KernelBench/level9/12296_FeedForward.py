import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.functional import dropout


class FeedForward(nn.Module):

    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = dropout(relu(self.fc1(x)))
        x = dropout(relu(self.fc2(x)))
        x = dropout(relu(self.fc3(x)))
        x = dropout(relu(self.fc4(x)))
        x = self.fc5(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]

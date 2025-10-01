import torch
from torch import nn


class Highway(nn.Module):

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc1.bias.data.fill_(-2)

    def forward(self, x):
        gate = self.sigmoid(self.fc1(x))
        return gate * self.relu(self.fc2(x)) + (1 - gate) * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]

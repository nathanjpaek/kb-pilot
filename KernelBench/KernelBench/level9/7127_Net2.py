import torch
from torch import nn


class Net2(nn.Module):
    """
    Net2 is a more complex network consisting of two hidden layers with 400
    and 300 neurons
    """
    hidden1 = 400
    hidden2 = 300

    def __init__(self, input_size):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, self.hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden2, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]

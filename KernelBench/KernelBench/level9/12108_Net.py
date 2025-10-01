import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, input_placeholder, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_placeholder, 255)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(255, 255)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(255, output_size)

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
    return [[], {'input_placeholder': 4, 'output_size': 4}]

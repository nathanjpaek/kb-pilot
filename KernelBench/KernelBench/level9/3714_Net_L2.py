import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_L2(nn.Module):

    def __init__(self, inputSize, kernel=64):
        super(Net_L2, self).__init__()
        self.inputSize = inputSize
        self.kernel = kernel
        self.fc1 = nn.Linear(self.inputSize, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)
        self.drop1 = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'inputSize': 4}]

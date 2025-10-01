import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):

    def __init__(self, num_classes=3):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = x.view(-1, 64 * 64 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 12288])]


def get_init_inputs():
    return [[], {}]

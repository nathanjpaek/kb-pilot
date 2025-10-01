import torch
import torch.nn as nn
import torch.utils.data
from math import *


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 2)
        self.fc4 = nn.Linear(2, 20)
        self.fc5 = nn.Linear(20, 400)
        self.fc6 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        z = self.fc3(x)
        return z

    def decode(self, z):
        z = self.relu(self.fc4(z))
        z = self.relu(self.fc5(z))
        return self.sigmoid(self.fc6(z))

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        x = self.decode(z)
        return x, z


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]

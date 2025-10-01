import torch
import torch.nn as nn


class Ecgclient(nn.Module):

    def __init__(self):
        super(Ecgclient, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]

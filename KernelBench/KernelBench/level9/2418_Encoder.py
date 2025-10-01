import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, 3, padding=1)
        self.maxp1 = torch.nn.MaxPool1d(2, padding=0)
        self.conv2 = torch.nn.Conv1d(64, 128, 3, padding=1)
        self.maxp2 = torch.nn.MaxPool1d(2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxp2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]

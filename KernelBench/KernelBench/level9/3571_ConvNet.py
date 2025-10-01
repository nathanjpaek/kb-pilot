import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 6, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 8, 3, 1)
        self.drp1 = nn.Dropout2d(0.25)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(288, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drp1(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 288)
        x = self.lin1(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

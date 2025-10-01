import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """Standard convolutional net for baseline
    Architecture: 2 convolutional layers, 3 fully connected layers.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        args = {'stride': 1, 'padding': 1}
        self.conv1 = nn.Conv2d(3, 10, 3, **args)
        self.conv2 = nn.Conv2d(10, 20, 3, **args)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1280, 640)
        self.fc2 = nn.Linear(640, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 1280)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

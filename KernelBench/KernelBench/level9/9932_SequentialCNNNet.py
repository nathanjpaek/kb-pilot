import torch
import torch.nn as nn


class SequentialCNNNet(nn.Module):

    def __init__(self):
        super(SequentialCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 84)
        self.fc3 = nn.Linear(84, 50)
        self.features = nn.Sequential(self.conv1, nn.ReLU(), self.pool,
            self.conv2, nn.ReLU(), self.pool)
        self.classifier = nn.Sequential(self.fc1, self.fc2, self.fc3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 5 * 5)
        x = self.classifier(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {}]

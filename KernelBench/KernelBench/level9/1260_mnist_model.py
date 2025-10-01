import torch
import torch.nn as nn
import torch.nn.functional as F


class mnist_model(nn.Module):

    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv3 = nn.Conv2d(64, 128, 1, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2(h1))
        h3 = self.pool1(self.relu(self.conv3(h2)))
        h3 = h3.view(-1, 128 * 14 * 14)
        h4 = self.fc1(h3)
        h5 = self.fc2(h4)
        return F.log_softmax(h5, dim=1)


def get_inputs():
    return [torch.rand([4, 1, 32, 32])]


def get_init_inputs():
    return [[], {}]

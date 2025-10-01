import torch
import torch.nn as nn
import torch.nn.functional as F


class neuralNet(nn.Module):

    def __init__(self):
        super(neuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.fc1 = nn.Linear(in_features=48 * 12 * 12, out_features=240)
        self.fc2 = nn.Linear(in_features=240, out_features=120)
        self.out = nn.Linear(in_features=120, out_features=17)

    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = t.reshape(-1, 48 * 12 * 12)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.out(t)
        return t


def get_inputs():
    return [torch.rand([4, 3, 256, 256])]


def get_init_inputs():
    return [[], {}]

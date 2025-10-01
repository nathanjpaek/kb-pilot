import torch
import torch.nn as nn
import torch.nn.functional as F


class SynthWide256(nn.Module):

    def __init__(self, num_c=10, f=1):
        super(SynthWide256, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32 * f, 3, padding=1)
        self.conv2 = nn.Conv2d(32 * f, 64 * f, 3, padding=1)
        self.conv3 = nn.Conv2d(64 * f, 128 * f, 3, padding=1)
        self.conv4 = nn.Conv2d(128 * f, 256 * f, 3, padding=1)
        self.conv5 = nn.Conv2d(256 * f, 512 * f, 3, padding=1)
        self.conv6 = nn.Conv2d(512 * f, 1024 * f, 3, padding=1)
        self.conv7 = nn.Conv2d(1024 * f, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, num_c)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = self.fc1(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 256, 256])]


def get_init_inputs():
    return [[], {}]

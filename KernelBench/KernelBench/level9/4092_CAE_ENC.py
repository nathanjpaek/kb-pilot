import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE_ENC(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 1000)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc1(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 96, 96])]


def get_init_inputs():
    return [[], {}]

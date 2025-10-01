import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimalBaselineNet(nn.Module):

    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(24 * 8 * 8, 128)
        self.cls = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 24 * 8 * 8)
        x = F.relu(self.fc(x))
        x = self.cls(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

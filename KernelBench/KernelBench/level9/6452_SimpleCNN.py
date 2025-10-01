import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self, num_channels, num_classes):
        super(SimpleCNN, self).__init__()
        C = num_channels
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=C * 8,
            kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=C * 8, out_channels=C * 32,
            kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=C * 32, out_channels=C * 128,
            kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(C * 128, 128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4, 'num_classes': 4}]

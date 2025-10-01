import torch
import torch.nn as nn


class QPCnet(nn.Module):

    def __init__(self, num_classes=2):
        super(QPCnet, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, 3, [1, 2], 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 2, [1, 2])
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 2, 64, 64])]


def get_init_inputs():
    return [[], {}]

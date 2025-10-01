import torch
from torch import nn
import torch.nn.functional as F


class mnistmodel_B(nn.Module):

    def __init__(self):
        super(mnistmodel_B, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=
            8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=5, stride=1)
        self.dense1 = nn.Linear(in_features=128, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

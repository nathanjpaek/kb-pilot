import torch
from torch import nn
import torch.nn.functional as F


class mnistmodel_A(nn.Module):

    def __init__(self):
        super(mnistmodel_A, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=
            5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =5, stride=2)
        self.dense1 = nn.Linear(in_features=64 * 12 * 12, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(x, 0.25)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

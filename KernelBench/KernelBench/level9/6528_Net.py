import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=
            (7, 3))
        self.pool = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size
            =(3, 3))
        self.fc1 = nn.Linear(7 * 8 * 20, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(7 * 8 * 20, -1)
        x = x.permute(1, 0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

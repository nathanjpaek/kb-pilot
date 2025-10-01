import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=
            5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size
            =3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size
            =3, padding=1)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Linear(in_features=1296, out_features=3)

    def forward(self, x):
        x = self.pooling(self.relu(self.conv1(x)))
        x = self.pooling(self.relu(self.conv2(x)))
        x = self.pooling(self.relu(self.conv2(x)))
        x = self.pooling(self.relu(self.conv3(x)))
        x = self.linear(x.view(-1, 1296))
        return x


def get_inputs():
    return [torch.rand([4, 3, 144, 144])]


def get_init_inputs():
    return [[], {}]

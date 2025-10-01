import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN class - defines model and forward operations"""

    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
            stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=
            3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size
            =3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size
            =3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Method override for forward operation
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = self.logsoftmax(x)
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

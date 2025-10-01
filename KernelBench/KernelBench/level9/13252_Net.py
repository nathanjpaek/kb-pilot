import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(17 * 17 * 64, 64)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 17 * 17 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return torch.sigmoid(self.fc2(x))


def get_inputs():
    return [torch.rand([4, 3, 288, 288])]


def get_init_inputs():
    return [[], {}]

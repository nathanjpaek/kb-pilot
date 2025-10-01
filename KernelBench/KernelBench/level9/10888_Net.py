import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, c1=32, c2=64, c3=128, c4=256, l1=512, d1=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(9, c1, (5, 5))
        self.conv2 = nn.Conv2d(c1, c2, (5, 5))
        self.conv3 = nn.Conv2d(c2, c3, (5, 5))
        self.conv4 = nn.Conv2d(c3, c4, (5, 5))
        x = torch.randn(900, 900).view(-1, 9, 100, 100)
        self.toLinear = -1
        self.convs(x)
        self.fc1 = nn.Linear(self.toLinear, l1)
        self.fc2 = nn.Linear(l1, 2)
        self.dropout1 = nn.Dropout(d1)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        if self.toLinear == -1:
            self.toLinear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.toLinear)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 9, 128, 128])]


def get_init_inputs():
    return [[], {}]

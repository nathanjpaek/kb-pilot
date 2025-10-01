import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3D(nn.Module):

    def __init__(self, cin, cout):
        super(Conv3D, self).__init__()
        self.conv1 = nn.Conv2d(cin, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(8, cout, 4, 3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cin': 4, 'cout': 4}]

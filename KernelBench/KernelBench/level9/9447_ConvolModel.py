import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolModel(nn.Module):

    def __init__(self):
        super(ConvolModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 2)
        self.conv2 = nn.Conv2d(5, 10, 2)
        self.conv3 = nn.Conv2d(10, 10, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.max_pool2d(self.conv3(x), 2)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

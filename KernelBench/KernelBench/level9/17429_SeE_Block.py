import torch
import torch.nn as nn
import torch.nn.functional as F


class SeE_Block(nn.Module):

    def __init__(self, channel):
        super(SeE_Block, self).__init__()
        self.channel = channel
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(self.channel, self.channel, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.channel, self.channel, 1, 1, 0)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2
            ), x.size(3)))
        fc1 = self.fc1(avg_pool)
        fc1 = self.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = self.sigmoid(fc2)
        see = x * fc2
        return see


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]

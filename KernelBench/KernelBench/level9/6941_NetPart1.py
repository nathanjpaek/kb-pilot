import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed


class NetPart1(nn.Module):

    def __init__(self):
        super(NetPart1, self).__init__()
        d1 = 768
        self.conv1 = nn.Conv2d(1, d1, 3, 1)
        self.conv2 = nn.Conv2d(d1, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

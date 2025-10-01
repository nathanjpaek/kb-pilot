import torch
from torch import nn
from torch.nn import functional as f


class TianzigeCNN(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1024, 5)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(1024, 256, 1, groups=8)
        self.conv3 = nn.Conv2d(256, dimension, 2, groups=16)
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.conv3.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = f.adaptive_avg_pool2d(x, output_size=1).squeeze()
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'dimension': 32}]

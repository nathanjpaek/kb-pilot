import torch
from torch import nn
from torch.nn import functional as f


class GlyphNet(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Linear(32, dimension)
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = f.adaptive_avg_pool2d(x, output_size=1).squeeze()
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'dimension': 4}]

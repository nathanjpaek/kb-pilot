import torch
import torch.nn as nn


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.GroupNorm(nChannels, nChannels, affine=True)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
            padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nChannels': 4, 'growthRate': 4}]

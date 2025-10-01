import torch
import torch.nn as nn
import torch.utils.data


class NetModel(nn.Module):

    def __init__(self, n1, n2):
        super(NetModel, self).__init__()
        self.layer1 = nn.Conv2d(1, n1, kernel_size=9, stride=1, padding=4,
            bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(n1, n2, kernel_size=5, stride=1, padding=2,
            bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Conv2d(n2, 1, kernel_size=5, stride=1, padding=2,
            bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'n1': 4, 'n2': 4}]

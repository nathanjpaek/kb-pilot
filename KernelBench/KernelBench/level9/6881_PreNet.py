import torch
import torch.nn as nn


class PreNet(nn.Module):

    def __init__(self):
        super(PreNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, labels):
        conv1 = self.act1(self.conv1(labels))
        conv2 = self.act2(self.conv2(conv1))
        return conv2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

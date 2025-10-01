import torch
import torch.nn as nn


class TinyCnn(nn.Module):

    def __init__(self, feature_extraction=False):
        super().__init__()
        self.feature_extraction = feature_extraction
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        if not self.feature_extraction:
            self.conv2 = nn.Conv2d(3, 10, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        if not self.feature_extraction:
            x = self.conv2(x)
            x = x.view(-1, 10)
        else:
            x = x.view(-1, 12)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
from abc import *


class Classifier(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_classes': 4}]

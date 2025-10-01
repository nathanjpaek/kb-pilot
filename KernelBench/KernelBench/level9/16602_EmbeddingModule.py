import torch
import torch.nn as nn


class EmbeddingModule(nn.Module):

    def __init__(self, in_channels, desc_channels):
        super(EmbeddingModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, desc_channels)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'desc_channels': 4}]

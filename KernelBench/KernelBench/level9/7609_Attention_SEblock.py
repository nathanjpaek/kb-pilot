import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_SEblock(nn.Module):

    def __init__(self, channels, reduction, temperature):
        super(Attention_SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.temperature = temperature
        self.channels = channels

    def forward(self, x):
        x = self.avg_pool(x).view(-1, self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.gumbel_softmax(x, tau=1, hard=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'reduction': 4, 'temperature': 4}]

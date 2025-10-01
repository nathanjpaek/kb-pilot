import torch
from torch import nn
from itertools import chain
import torch.nn.functional as F


class VeryFlatNet(nn.Module):

    def __init__(self, num_channels=128, kernel_size=9):
        super(VeryFlatNet, self).__init__()
        self.num_channels = num_channels
        None
        padding = int((kernel_size - 1) / 2)
        self.convfeatures = nn.Conv2d(1, num_channels, groups=1,
            kernel_size=kernel_size, padding=padding, stride=1)
        channels = 1 + num_channels
        self.convp0 = nn.Conv2d(channels, channels // 2, kernel_size=1,
            padding=0)
        channels = channels // 2
        self.convp1 = nn.Conv2d(channels, channels // 2, kernel_size=1,
            padding=0)
        channels = channels // 2
        self.convp2 = nn.Conv2d(channels, channels // 2, kernel_size=1,
            padding=0)
        channels = channels // 2
        self.convpf = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def set_weights(self, weights, bias=0):
        next(self.parameters()).device
        with torch.no_grad():
            length = weights.shape[0]
            None
            self.convfeatures._parameters['weight'][0:length
                ] = torch.from_numpy(weights)

    def lastparameters(self):
        return chain(self.convp0.parameters(), self.convp1.parameters(),
            self.convp2.parameters(), self.convpf.parameters())

    def verylastparameters(self):
        return self.convp5.parameters()

    def forward(self, x):
        y = self.convfeatures(x)
        features = F.relu(torch.cat((x, y), 1))
        features = F.relu(self.convp0(features))
        features = F.relu(self.convp1(features))
        features = F.relu(self.convp2(features))
        prediction = self.convpf(features)
        return prediction


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

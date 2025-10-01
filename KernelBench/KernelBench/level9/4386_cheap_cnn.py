import torch
import torch.nn as nn
import torch.nn.functional as F


class cheap_cnn(nn.Module):

    def __init__(self):
        super(cheap_cnn, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x.size(0)
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        return self.flatten(x)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

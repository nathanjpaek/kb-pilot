import torch
import torch.nn as nn
import torch.utils.data


class FociDetector(nn.Module):

    def __init__(self, input_channels=3, input_size=17, ksize=5,
        hidden_channels=10):
        super(FociDetector, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, ksize,
            stride=2, padding=int((ksize - 1) / 2))
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, ksize,
            stride=2, padding=int((ksize - 1) / 2))
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, ksize,
            stride=2, padding=int((ksize - 1) / 2))
        self.finalmapsize = ksize
        self.convf = nn.Conv2d(hidden_channels, 1, self.finalmapsize,
            padding=int((ksize - 1) / 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.conv1(x))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = self.convf(output)
        return output


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

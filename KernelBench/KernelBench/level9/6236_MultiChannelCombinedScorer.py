import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


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


class MultiChannelCombinedScorer(nn.Module):

    def __init__(self, input_size=17, ksize=5, hidden_channels=10):
        super(MultiChannelCombinedScorer, self).__init__()
        self.channel1 = FociDetector(input_channels=1, input_size=
            input_size, ksize=ksize, hidden_channels=hidden_channels)
        self.channel2 = FociDetector(input_channels=1, input_size=
            input_size, ksize=ksize, hidden_channels=hidden_channels)

    def forward(self, x):
        output1 = torch.sigmoid(F.interpolate(self.channel1(x[:, [0], :, :]
            ), size=(x.shape[2], x.shape[3])))
        output2 = torch.sigmoid(F.interpolate(self.channel2(x[:, [1], :, :]
            ), size=(x.shape[2], x.shape[3])))
        output3 = torch.sigmoid(x[:, [0], :, :])
        output4 = torch.sigmoid(x[:, [1], :, :])
        return output1 * output2 * output3 * output4

    def forward_vis(self, x):
        output1 = torch.sigmoid(F.interpolate(self.channel1(x[:, [0], :, :]
            ), size=(x.shape[2], x.shape[3])))
        output2 = torch.sigmoid(F.interpolate(self.channel2(x[:, [1], :, :]
            ), size=(x.shape[2], x.shape[3])))
        output3 = torch.sigmoid(x[:, [0], :, :])
        output4 = torch.sigmoid(x[:, [1], :, :])
        return output1, output2, output3, output4


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

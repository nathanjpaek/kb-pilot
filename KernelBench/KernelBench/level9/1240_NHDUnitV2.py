import torch
import torch.nn as nn


class NHDUnitV2(nn.Module):

    def __init__(self, in_channels, hidden_channels, *args, **kwargs):
        super(NHDUnitV2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self._build()

    def _build(self):
        self.conv_1 = nn.Conv2d(self.in_channels, self.hidden_channels, 1)
        self.conv_2 = nn.Conv2d(self.hidden_channels, self.hidden_channels,
            3, stride=1, padding=1)
        self.conv_trans_1 = nn.Conv2d(self.hidden_channels, self.
            hidden_channels, 1)
        self.conv_trans_2 = nn.Conv2d(self.hidden_channels, 1, 1)
        self.conv_atmos = nn.Conv2d(self.hidden_channels, self.in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inp):
        feat = self.relu(self.conv_2(self.relu(self.conv_1(inp))))
        trans = self.sigmoid(self.conv_trans_2(self.relu(self.conv_trans_1(
            feat))))
        atmos = self.sigmoid(self.conv_atmos(self.global_avg(feat)))
        out = inp * trans + (1 - trans) * atmos
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'hidden_channels': 4}]

import torch
import torch.nn as nn


class SAC(nn.Module):

    def __init__(self, input_channel, out_channel):
        super(SAC, self).__init__()
        self.conv_1 = nn.Conv3d(input_channel, out_channel, kernel_size=3,
            stride=1, padding=1)
        self.conv_3 = nn.Conv3d(input_channel, out_channel, kernel_size=3,
            stride=1, padding=2, dilation=2)
        self.conv_5 = nn.Conv3d(input_channel, out_channel, kernel_size=3,
            stride=1, padding=3, dilation=3)
        self.weights = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(0)

    def forward(self, inputs):
        feat_1 = self.conv_1(inputs)
        feat_3 = self.conv_3(inputs)
        feat_5 = self.conv_5(inputs)
        weights = self.softmax(self.weights)
        feat = feat_1 * weights[0] + feat_3 * weights[1] + feat_5 * weights[2]
        return feat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'out_channel': 4}]

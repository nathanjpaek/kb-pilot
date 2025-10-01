import torch
from torch import nn


class ChannelAttention_a(nn.Module):

    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention_a, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(ratio, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, _, _ = x.size()
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        y1 = out[:, 0:1, :, :]
        y1 = y1.expand(b, 32, 1, 1)
        y2 = out[:, 1:, :, :]
        y2 = y2.expand(b, 32, 1, 1)
        y_sum = torch.cat((y1, y2), 1)
        return y_sum


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4}]

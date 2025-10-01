import torch
from torch import nn


class channel_attention(nn.Module):

    def __init__(self, in_channels, feature_size):
        super(channel_attention, self).__init__()
        self.fc1 = nn.Linear(feature_size * feature_size, feature_size,
            bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(in_channels))
        self.softmax = nn.Softmax()

    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)
        c_f = self.fc1(target_feature_resize)
        c_f = self.relu1(c_f)
        c_f = self.fc2(c_f)
        c_f = c_f.view(b, c)
        channel_attention_weight = self.softmax(c_f + self.bias)
        return channel_attention_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'feature_size': 4}]

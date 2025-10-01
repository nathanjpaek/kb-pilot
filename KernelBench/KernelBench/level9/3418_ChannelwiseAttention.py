import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseAttention(nn.Module):

    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()
        self.in_channels = in_channels
        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, _h, _w = input_.size()
        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))
        ca_act_reg = torch.mean(feats)
        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()
        return feats, ca_act_reg


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]

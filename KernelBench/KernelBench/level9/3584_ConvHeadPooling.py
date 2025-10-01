import torch
import torch.nn as nn
from typing import Tuple


class ConvHeadPooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(ConvHeadPooling, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride +
            1, padding=stride // 2, stride=stride, padding_mode=
            padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token) ->Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4, 'stride': 1}]

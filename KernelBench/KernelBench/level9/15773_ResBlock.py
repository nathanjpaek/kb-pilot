import torch
import torch.nn as nn
from typing import Tuple


def conv3x3(in_channels: 'int', out_channels: 'int', stride: 'int'=1,
    padding: 'int'=1) ->nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=padding, bias=True)
    nn.init.xavier_normal_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.fill_(0)
    return conv


class ResBlock(nn.Module):

    def __init__(self, features: 'int'):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(features, features)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(features, features)
        self.relu2 = nn.ReLU()

    def forward(self, activated_input: 'torch.Tensor', residual_input:
        'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(activated_input)
        x = self.relu1(x)
        x = self.conv2(x)
        residual_output = x + residual_input
        activated_output = self.relu2(residual_output)
        return activated_output, residual_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]

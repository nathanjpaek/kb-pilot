import torch
from typing import Tuple
from torch import nn


class _ImpalaResBlock(nn.Module):

    def __init__(self, n_channels: 'int'):
        super().__init__()
        self.n_channels = n_channels
        kernel_size = 3
        padding = 1
        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding
            =padding)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding
            =padding)

    def forward(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        x = self.relu_inplace(x)
        x = self.conv2(x)
        x += inputs
        return x


class _ImpalaBlock(nn.Module):

    def __init__(self, n_channels_in: 'int', n_channels_out: 'int'):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        kernel_size = 3
        padding = 1
        self.conv1 = nn.Conv2d(n_channels_in, n_channels_out, kernel_size,
            padding=padding)
        self.pool = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.res1 = _ImpalaResBlock(n_channels_out)
        self.res2 = _ImpalaResBlock(n_channels_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class _ImpalaCNN(nn.Module):

    def __init__(self, img_shape: 'Tuple[int, int, int]', n_extra_feats:
        'int'=0, n_outputs: 'int'=256):
        super().__init__()
        self.n_outputs = n_outputs
        h, w, c = img_shape
        self.block1 = _ImpalaBlock(c, 16)
        self.block2 = _ImpalaBlock(16, 32)
        self.block3 = _ImpalaBlock(32, 32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        test_img = torch.empty(c, h, w)[None]
        n_feats = self.block3(self.block2(self.block1(test_img))).numel()
        self.linear = nn.Linear(n_feats + n_extra_feats, self.n_outputs)

    def forward(self, x, extra_obs=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = self.flatten(x)
        if extra_obs is not None:
            x = torch.cat((x, extra_obs), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'img_shape': [4, 4, 4]}]

import torch
import torch.nn as nn


class cSEmodule(nn.Module):
    """ SpatialSequeezeExcitationModule
        input: [B, C, H, W] torch tensor
        output: [B, C, H, W] torch tensor
    """

    def __init__(self, in_channel):
        super().__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.down_linear = nn.Linear(in_channel, in_channel // 2)
        self.up_linear = nn.Linear(in_channel // 2, in_channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _h, _w = x.shape
        skip_connection = x
        x = self.global_avg(x)
        x = self.flatten(x)
        x = self.down_linear(x)
        x = self.relu(x)
        x = self.up_linear(x)
        x = self.sigmoid(x)
        x = x.reshape(b, c, 1, 1)
        x = x * skip_connection
        return x


class sSEmodule(nn.Module):
    """ ChannelSequeezeExcitationModule
        input: [B, C, H, W] torch tensor
        output: [B, C, H, W] torch tensor
    """

    def __init__(self, in_channel):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connection = x
        x = self.conv2d(x)
        x = self.sigmoid(x)
        None
        x = x * skip_connection
        return x


class scSEmodule(nn.Module):
    """ ConcurrentSpatialChannelSequeezeExcitationModule
        input: [B, C, H, W] torch tensor
        output: [B, C, H, W] torch tensor
    """

    def __init__(self, in_channel):
        super().__init__()
        self.cSEmodule = cSEmodule(in_channel=in_channel)
        self.sSEmodule = sSEmodule(in_channel=in_channel)

    def forward(self, x):
        cse_branch = self.cSEmodule(x)
        sse_branch = self.sSEmodule(x)
        return torch.max(cse_branch, sse_branch)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4}]

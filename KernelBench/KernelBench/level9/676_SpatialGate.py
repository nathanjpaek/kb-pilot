import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class Conv3DSimple(nn.Conv3d):

    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3):
        padding = (kernel_size - 1) // 2
        super(Conv3DSimple, self).__init__(in_channels=in_planes,
            out_channels=out_planes, kernel_size=(3, kernel_size,
            kernel_size), stride=(1, stride, stride), padding=(1, padding,
            padding), bias=False)


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1)
            .unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self, conv_builder=Conv3DSimple):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = conv_builder(2, 1, kernel_size=7)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale, scale


def get_inputs():
    return [torch.rand([4, 2, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]

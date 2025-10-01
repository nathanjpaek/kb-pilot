import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed


class conv(nn.Module):

    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=
            kernel_size, stride=stride, padding=(self.kernel_size - 1) // 2,
            padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=
            False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):

    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale,
            align_corners=True, mode='bilinear')
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {'num_in_layers': 1, 'num_out_layers': 1, 'kernel_size': 4,
        'scale': 1.0}]

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class avgpool(nn.Module):
    """
    Mean pooling class - downsampling
    """

    def __init__(self, up_size=0):
        super(avgpool, self).__init__()

    def forward(self, x):
        out_man = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1:
            :2] + x[:, :, 1::2, 1::2]) / 4
        return out_man


class ResidualBlock(nn.Module):
    """
    Residual block class
    3 types: upsample, downsample, None 
    """

    def __init__(self, in_dim, out_dim, resample=None, up_size=0):
        super(ResidualBlock, self).__init__()
        if resample == 'up':
            self.bn1 = nn.BatchNorm2d(in_dim)
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.upsample = torch.nn.Upsample(up_size, 2)
            self.upsample = torch.nn.Upsample(scale_factor=2)
            self.upsample_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.bn2 = nn.BatchNorm2d(out_dim)
        elif resample == 'down':
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.pool = avgpool()
            self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
        elif resample is None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
        self.resample = resample

    def forward(self, x):
        if self.resample is None:
            shortcut = x
            output = x
            output = nn.functional.relu(output)
            output = self.conv1(output)
            output = nn.functional.relu(output)
            output = self.conv2(output)
        elif self.resample == 'up':
            shortcut = x
            output = x
            shortcut = self.upsample(shortcut)
            shortcut = self.upsample_conv(shortcut)
            output = self.bn1(output)
            output = nn.functional.relu(output)
            output = self.conv1(output)
            output = self.bn2(output)
            output = nn.functional.relu(output)
            output = self.upsample(output)
            output = self.conv2(output)
        elif self.resample == 'down':
            shortcut = x
            output = x
            shortcut = self.pool_conv(shortcut)
            shortcut = self.pool(shortcut)
            output = nn.functional.relu(output)
            output = self.conv1(output)
            output = nn.functional.relu(output)
            output = self.conv2(output)
            output = self.pool(output)
        return output + shortcut


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]

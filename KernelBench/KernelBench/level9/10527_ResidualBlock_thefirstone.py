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


class ResidualBlock_thefirstone(nn.Module):
    """
    First residual block class 
    """

    def __init__(self, in_dim, out_dim, resample=None, up_size=0):
        super(ResidualBlock_thefirstone, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
        self.pool = avgpool()
        self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        shortcut = x
        output = x
        shortcut = self.pool(shortcut)
        shortcut = self.pool_conv(shortcut)
        output = self.conv1(output)
        output = nn.functional.relu(output)
        output = self.conv2(output)
        output = self.pool(output)
        return output + shortcut


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]

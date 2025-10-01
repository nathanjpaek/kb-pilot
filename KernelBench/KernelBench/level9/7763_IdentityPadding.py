import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.nn.functional as F


class IdentityPadding(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityPadding, self).__init__()
        if stride == 2:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True
                )
        else:
            self.pooling = None
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        if self.pooling is not None:
            out = self.pooling(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]

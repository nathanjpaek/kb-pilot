import torch
import torch.nn as nn


class vggUpconv(nn.Module):
    """Some Information about vggUpconv"""

    def __init__(self, in_ch, out_ch, upsample=True):
        super(vggUpconv, self).__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.upsample = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = self.conv1(x1)
        sum = x1 + x2
        return sum


def get_inputs():
    return [torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]

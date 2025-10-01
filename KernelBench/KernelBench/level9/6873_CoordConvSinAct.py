import torch
import torch.nn as nn


class SinAct(nn.Module):

    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class CoordConvSinAct(nn.Module):
    """
  Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """

    def __init__(self, in_channels, out_channels, channels_per_group=16, **
        kwargs):
        super().__init__()
        self.coord_conv = nn.Conv2d(2, out_channels, **kwargs)
        self.sin_act = SinAct()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        pass

    def forward(self, input):
        batch, _, H, W = input.shape
        x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=input.device),
            torch.linspace(-1, 1, H, device=input.device))
        x = x.T
        y = y.T
        xy = torch.stack((x, y), dim=0)
        xy = xy.expand((batch, -1, -1, -1))
        xy_fea = self.coord_conv(xy)
        xy_fea = self.sin_act(xy_fea)
        out = self.conv(input)
        out = xy_fea + out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]

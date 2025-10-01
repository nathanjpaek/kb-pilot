import torch
import torch.nn as nn
import torch.nn.functional as F


class PA(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


class PA_UP(nn.Module):

    def __init__(self, nf, unf, out_nc, scale=4):
        super(PA_UP, self).__init__()
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        if scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, fea):
        fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att2(fea))
        fea = self.lrelu(self.HRconv2(fea))
        fea = self.conv_last(fea)
        return fea


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nf': 4, 'unf': 4, 'out_nc': 4}]

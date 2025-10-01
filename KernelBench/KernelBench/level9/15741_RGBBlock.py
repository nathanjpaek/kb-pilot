import torch
from torch import nn
import torch.nn.functional as F


class Conv2DMod(nn.Module):

    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1,
        dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel,
            kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, _c, h, w = x.shape
        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)
        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) +
                EPS)
            weights = weights * d
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)
        padding = self._get_same_padding(h, self.kernel, self.dilation,
            self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.filters, h, w)
        return x


class RGBBlock(nn.Module):

    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)
        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        x = self.conv(x, style)
        if prev_rgb is not None:
            x = x + prev_rgb
        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def forward_(self, x, prev_rgb, style):
        x = self.conv(x, style)
        if prev_rgb is not None:
            x = x + prev_rgb
        if self.upsample is not None:
            x = self.upsample(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4, 'input_channel': 4, 'upsample': 4}]

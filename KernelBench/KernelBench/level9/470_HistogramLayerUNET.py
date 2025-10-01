import torch
import numpy as np
import torch.nn as nn


class HistogramLayerUNET(nn.Module):

    def __init__(self, in_channels, kernel_size, dim=2, num_bins=4, stride=
        None, padding=0, normalize_count=True, normalize_bins=True,
        count_include_pad=False, ceil_mode=False, skip_connection=False):
        super(HistogramLayerUNET, self).__init__()
        self.in_channels = in_channels
        self.numBins = num_bins
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        self.skip_connection = skip_connection
        if self.dim == 1:
            self.bin_centers_conv = nn.Conv1d(self.in_channels, self.
                numBins * self.in_channels, 1, groups=self.in_channels,
                bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv1d(self.numBins * self.
                in_channels, self.numBins * self.in_channels, 1, groups=
                self.numBins * self.in_channels, bias=False)
            self.hist_pool = nn.AvgPool1d(self.filt_dim, stride=self.stride,
                padding=self.padding, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        elif self.dim == 2:
            self.bin_centers_conv = nn.Conv2d(self.in_channels, self.
                numBins * self.in_channels, 1, groups=self.in_channels,
                bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv2d(self.numBins * self.
                in_channels, self.numBins * self.in_channels, 1, groups=
                self.numBins * self.in_channels, bias=False)
            self.hist_pool = nn.AvgPool2d(self.kernel_size, stride=self.
                stride, padding=self.padding, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        elif self.dim == 3:
            self.bin_centers_conv = nn.Conv3d(self.in_channels, self.
                numBins * self.in_channels, 1, groups=self.in_channels,
                bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv3d(self.numBins * self.
                in_channels, self.numBins * self.in_channels, 1, groups=
                self.numBins * self.in_channels, bias=False)
            self.hist_pool = nn.AvgPool3d(self.filt_dim, stride=self.stride,
                padding=self.padding, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        else:
            raise RuntimeError('Invalid dimension for histogram layer')

    def forward(self, xx):
        xx = self.bin_centers_conv(xx)
        xx = self.bin_widths_conv(xx)
        xx = torch.exp(-xx ** 2)
        if self.normalize_bins:
            xx = self.constrain_bins(xx)
        if not self.skip_connection:
            if self.normalize_count:
                xx = self.hist_pool(xx)
            else:
                xx = np.prod(np.asarray(self.hist_pool.kernel_size)
                    ) * self.hist_pool(xx)
        else:
            pass
        return xx

    def constrain_bins(self, xx):
        if self.dim == 1:
            n, c, l = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, l).sum(2
                ) + torch.tensor(1e-05)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum
        elif self.dim == 2:
            n, c, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2
                ) + torch.tensor(1e-05)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum
        elif self.dim == 3:
            n, c, d, h, w = xx.size()
            xx_sum = xx.reshape(n, c // self.numBins, self.numBins, d, h, w
                ).sum(2) + torch.tensor(1e-05)
            xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
            xx = xx / xx_sum
        else:
            raise RuntimeError('Invalid dimension for histogram layer')
        return xx


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'kernel_size': 4}]

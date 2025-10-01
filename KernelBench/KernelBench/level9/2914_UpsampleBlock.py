import torch
import torch.nn as nn


class PixelShuffle1d(nn.Module):

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]
        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width
        x = x.contiguous().view([batch_size, self.upscale_factor,
            long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        return x


class SamePaddingConv1d(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=int((
            kernel_size - 1) / 2))

    def forward(self, x):
        return self.conv(x)


class UpConv1d(nn.Module):

    def __init__(self, in_dim, out_dim, scale_factor, kernel_size):
        super().__init__()
        self.pixel_shuffer = PixelShuffle1d(scale_factor)
        self.conv = SamePaddingConv1d(in_dim // scale_factor, out_dim,
            kernel_size)

    def forward(self, x):
        x = self.pixel_shuffer(x)
        return self.conv(x)


class UpsampleBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.deconv = UpConv1d(in_dim, out_dim, scale_factor=4, kernel_size=1)
        self.conv_1 = UpConv1d(in_dim, out_dim, scale_factor=4, kernel_size=25)
        self.conv_2 = nn.Conv1d(out_dim, out_dim, 25, padding=12)
        self.LReLU_1 = nn.LeakyReLU(0.2)
        self.LReLU_2 = nn.LeakyReLU(0.2)

    def forward(self, input):
        shortcut = self.deconv(input)
        x = input
        x = self.conv_1(x)
        x = self.LReLU_1(x)
        x = self.conv_2(x)
        x = self.LReLU_2(x)
        return x + shortcut


def get_inputs():
    return [torch.rand([4, 4, 1, 1])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]

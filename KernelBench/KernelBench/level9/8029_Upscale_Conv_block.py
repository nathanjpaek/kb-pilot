import torch
from torch import nn


class ConvShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=
        'same', upscale_factor=2, padding_mode='zeros'):
        super(ConvShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor **
            2, kernel_size, padding=padding, padding_mode=padding_mode)

    def forward(self, X):
        X = self.conv(X)
        return nn.functional.pixel_shuffle(X, self.upscale_factor)


class Upscale_Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=
        'same', upscale_factor=2, main_upscale=ConvShuffle, shortcut=
        ConvShuffle, padding_mode='zeros', activation=nn.functional.relu,
        Ch=0, Cw=0):
        assert isinstance(upscale_factor, int)
        super(Upscale_Conv_block, self).__init__()
        self.shortcut = shortcut(in_channels, out_channels, kernel_size,
            padding=padding, padding_mode=padding_mode, upscale_factor=
            upscale_factor)
        self.activation = activation
        self.upscale = main_upscale(in_channels, out_channels, kernel_size,
            padding=padding, padding_mode=padding_mode, upscale_factor=
            upscale_factor)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size,
            padding=padding, padding_mode=padding_mode)
        self.Ch = Ch
        self.Cw = Cw

    def forward(self, X):
        X_shortcut = self.shortcut(X)
        X = self.activation(X)
        X = self.upscale(X)
        X = self.activation(X)
        X = self.conv(X)
        H, W = X.shape[2:]
        H2, W2 = X_shortcut.shape[2:]
        if H2 > H or W2 > W:
            padding_height = (H2 - H) // 2
            padding_width = (W2 - W) // 2
            X = X + X_shortcut[:, :, padding_height:padding_height + H,
                padding_width:padding_width + W]
        else:
            X = X + X_shortcut
        return X[:, :, self.Ch:, self.Cw:]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]

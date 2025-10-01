import torch
import torch.nn as nn


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)
    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels,
            scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels,
            out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class ConvNorm(nn.Module):

    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size
            =kernel_size, bias=True)
        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class PixelShuffle(nn.Module):

    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


class UpConvNorm(nn.Module):

    def __init__(self, in_channels, out_channels, mode='transpose', norm=False
        ):
        super(UpConvNorm, self).__init__()
        if mode == 'transpose':
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size=4, stride=2, padding=1)
        elif mode == 'shuffle':
            self.upconv = nn.Sequential(ConvNorm(in_channels, 4 *
                out_channels, kernel_size=3, stride=1, norm=norm),
                PixelShuffle(2))
        else:
            self.upconv = nn.Sequential(nn.Upsample(mode='nearest',
                scale_factor=2, align_corners=False), ConvNorm(in_channels,
                out_channels, kernel_size=1, stride=1, norm=norm))

    def forward(self, x):
        out = self.upconv(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]

import torch
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class baseline_upscale(nn.Module):

    def __init__(self, nf):
        super(baseline_upscale, self).__init__()
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.HR_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.last_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        initialize_weights([self.upconv1, self.upconv2, self.HR_conv, self.
            last_conv], 0.1)

    def forward(self, x):
        x = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
        x = self.lrelu(self.pixel_shuffle(self.upconv2(x)))
        x = self.last_conv(self.lrelu(self.HR_conv(x)))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nf': 4}]

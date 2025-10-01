import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):

    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
        fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        elif with_conv:
            self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                kernel=3, down=True, resample_kernel=fir_kernel, use_bias=
                True, kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        _B, _C, _H, _W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        elif not self.with_conv:
            x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
        else:
            x = self.Conv2d_0(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn.parallel
import torch.utils.data
from torch import nn
import torch.fft


class LossyYCbCr(nn.Module):

    def forward(self, rgb: 'torch.Tensor'):
        return torch.cat([0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 *
            rgb[:, 2:3], -0.16875 * rgb[:, 0:1] - 0.33126 * rgb[:, 1:2] + 
            0.5 * rgb[:, 2:3], 0.5 * rgb[:, 0:1] - 0.41869 * rgb[:, 1:2] - 
            0.08131 * rgb[:, 2:3]], dim=1)


class LossyRGB(nn.Module):

    def forward(self, ycbcr: 'torch.Tensor'):
        return torch.cat([ycbcr[:, 0:1] + 1.402 * ycbcr[:, 2:3], ycbcr[:, 0
            :1] - 0.34413 * ycbcr[:, 1:2] - 0.71414 * ycbcr[:, 2:3], ycbcr[
            :, 0:1] + 1.772 * ycbcr[:, 1:2]], dim=1)


class LosslessYCbCr(nn.Module):

    def forward(self, rgb: 'torch.Tensor'):
        return torch.cat([(rgb[:, 0:1] + 2 * rgb[:, 1:2] + rgb[:, 2:3]) / 4,
            rgb[:, 2:3] - rgb[:, 1:2], rgb[:, 0:1] - rgb[:, 1:2]], dim=1)


class LosslessRGB(nn.Module):

    def forward(self, ycbcr: 'torch.Tensor'):
        return torch.cat([ycbcr[:, 2:3] + ycbcr[:, 0:1] - 0.25 * ycbcr[:, 1
            :2] - 0.25 * ycbcr[:, 2:3], ycbcr[:, 0:1] - 0.25 * ycbcr[:, 1:2
            ] - 0.25 * ycbcr[:, 2:3], ycbcr[:, 1:2] + ycbcr[:, 0:1] - 0.25 *
            ycbcr[:, 1:2] - 0.25 * ycbcr[:, 2:3]], dim=1)


class DWT(nn.Module):

    def __init__(self, lossy: 'bool'=True):
        super().__init__()
        if lossy:
            dec_lo = [0.02674875741080976, -0.01686411844287495, -
                0.07822326652898785, 0.2668641184428723, 0.6029490182363579,
                0.2668641184428723, -0.07822326652898785, -
                0.01686411844287495, 0.02674875741080976]
            self.to_ycbcr = LossyYCbCr()
            self.to_rgb = LossyRGB()
            None
        else:
            dec_lo = [-0.125, 0.25, 0.75, 0.25, -0.125]
            self.to_ycbcr = LosslessYCbCr()
            self.to_rgb = LosslessRGB()
            None
        self.dwt_vertical = nn.Conv2d(3, 3, (len(dec_lo), 1), padding=(len(
            dec_lo) // 2, 0), bias=False, padding_mode='reflect')
        self.dwt_horizontal = nn.Conv2d(3, 3, (1, len(dec_lo)), padding=(0,
            len(dec_lo) // 2), bias=False, padding_mode='reflect')
        self.dwt_vertical.weight.requires_grad = False
        self.dwt_horizontal.weight.requires_grad = False
        self.dwt_vertical.weight.fill_(0)
        self.dwt_horizontal.weight.fill_(0)
        for c in range(3):
            for i in range(len(dec_lo)):
                self.dwt_vertical.weight[c, c, i, 0] = dec_lo[i]
                self.dwt_horizontal.weight[c, c, 0, i] = dec_lo[i]

    def forward(self, image: 'torch.Tensor', k: 'int'=1) ->torch.Tensor:
        """
        Args:
            image: 画素値0.0-1.0の画像バッチ
        """
        ll = self.to_ycbcr(image)
        for i in range(k):
            ll = self.dwt_vertical(self.dwt_horizontal(ll))
        rgb_shifted = self.to_rgb(ll)
        return rgb_shifted


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]

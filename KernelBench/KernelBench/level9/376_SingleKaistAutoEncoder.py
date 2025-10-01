import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(module, mode='fan_out', nonlinearity='relu', bias=0,
    distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=
            nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=
            nonlinearity)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class SingleKaistAutoEncoder(nn.Module):

    def __init__(self):
        super(SingleKaistAutoEncoder, self).__init__()
        self.conv1_rgb = nn.Conv2d(3, 32, 5, 4, 2)
        self.conv2_rgb = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3_rgb = nn.Conv2d(64, 256, 3, 2, 1)
        self.de_conv1_rgb = nn.ConvTranspose2d(256, 64, kernel_size=(4, 4),
            stride=(2, 2), padding=(1, 1))
        self.de_conv2_rgb = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4),
            stride=(2, 2), padding=(1, 1))
        self.de_conv3_rgb = nn.ConvTranspose2d(32, 3, kernel_size=(6, 6),
            stride=(4, 4), padding=(1, 1))

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        """
        return a bilinear filter tensor
        """
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center
            ) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size,
            kernel_size), dtype='float32')
        for i in range(in_channels):
            for j in range(out_channels):
                weight[i, j, :, :] = filt
        return torch.from_numpy(weight)

    def init_weights(self):
        kaiming_init(self.conv1_rgb)
        kaiming_init(self.conv2_rgb)
        kaiming_init(self.conv3_rgb)
        self.de_conv1_rgb.weight.data = self.bilinear_kernel(self.
            de_conv1_rgb.in_channels, self.de_conv1_rgb.out_channels, 4)
        self.de_conv2_rgb.weight.data = self.bilinear_kernel(self.
            de_conv2_rgb.in_channels, self.de_conv2_rgb.out_channels, 4)
        self.de_conv3_rgb.weight.data = self.bilinear_kernel(self.
            de_conv3_rgb.in_channels, self.de_conv3_rgb.out_channels, 6)

    def forward(self, img_rgb):
        rgb_down_2x = F.relu(self.conv1_rgb(img_rgb), True)
        rgb_down_4x = F.relu(self.conv2_rgb(rgb_down_2x), True)
        code = F.relu(self.conv3_rgb(rgb_down_4x), True)
        rgb_up_2x = F.relu(self.de_conv1_rgb(code), True)
        rgb_up_4x = F.relu(self.de_conv2_rgb(rgb_up_2x), True)
        decode = torch.tanh(self.de_conv3_rgb(rgb_up_4x))
        return code, decode


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

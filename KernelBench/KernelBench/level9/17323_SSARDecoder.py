import torch
from torch import nn


class SSARDecoder(nn.Module):

    def __init__(self):
        super(SSARDecoder, self).__init__()
        self.deconv0 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(16, 8, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(8, 2, 4, 2, (2, 1))

    def forward(self, x):
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        mask = self.deconv4(x)
        return mask


def get_inputs():
    return [torch.rand([4, 256, 4, 4])]


def get_init_inputs():
    return [[], {}]

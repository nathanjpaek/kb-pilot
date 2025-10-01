import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


def concat(a, b):
    return torch.cat((a, b), 1)


def pool(x):
    return F.max_pool2d(x, 2, 2)


def relu(x):
    return F.relu(x, inplace=True)


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode='nearest')


class Autoencoder(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(Autoencoder, self).__init__()
        ic = in_channels
        ec1 = 32
        ec2 = 48
        ec3 = 64
        ec4 = 80
        ec5 = 112
        dc5 = 160
        dc4 = 112
        dc3 = 96
        dc2 = 64
        dc1a = 64
        dc1b = 32
        oc = out_channels
        self.enc_conv0 = Conv(ic, ec1)
        self.enc_conv1 = Conv(ec1, ec1)
        self.enc_conv2 = Conv(ec1, ec2)
        self.enc_conv3 = Conv(ec2, ec3)
        self.enc_conv4 = Conv(ec3, ec4)
        self.enc_conv5 = Conv(ec4, ec5)
        self.dec_conv5a = Conv(ec5 + ec4, dc5)
        self.dec_conv5b = Conv(dc5, dc5)
        self.dec_conv4a = Conv(dc5 + ec3, dc4)
        self.dec_conv4b = Conv(dc4, dc4)
        self.dec_conv3a = Conv(dc4 + ec2, dc3)
        self.dec_conv3b = Conv(dc3, dc3)
        self.dec_conv2a = Conv(dc3 + ec1, dc2)
        self.dec_conv2b = Conv(dc2, dc2)
        self.dec_conv1a = Conv(dc2 + ic, dc1a)
        self.dec_conv1b = Conv(dc1a, dc1b)
        self.dec_conv0 = Conv(dc1b, oc)

    def forward(self, input):
        x = relu(self.enc_conv0(input))
        x = relu(self.enc_conv1(x))
        x = pool1 = pool(x)
        x = relu(self.enc_conv2(x))
        x = pool2 = pool(x)
        x = relu(self.enc_conv3(x))
        x = pool3 = pool(x)
        x = relu(self.enc_conv4(x))
        x = pool4 = pool(x)
        x = relu(self.enc_conv5(x))
        x = pool(x)
        x = upsample(x)
        x = concat(x, pool4)
        x = relu(self.dec_conv5a(x))
        x = relu(self.dec_conv5b(x))
        x = upsample(x)
        x = concat(x, pool3)
        x = relu(self.dec_conv4a(x))
        x = relu(self.dec_conv4b(x))
        x = upsample(x)
        x = concat(x, pool2)
        x = relu(self.dec_conv3a(x))
        x = relu(self.dec_conv3b(x))
        x = upsample(x)
        x = concat(x, pool1)
        x = relu(self.dec_conv2a(x))
        x = relu(self.dec_conv2b(x))
        x = upsample(x)
        x = concat(x, input)
        x = relu(self.dec_conv1a(x))
        x = relu(self.dec_conv1b(x))
        x = self.dec_conv0(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]

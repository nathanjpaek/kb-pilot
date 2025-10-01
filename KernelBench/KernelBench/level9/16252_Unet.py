import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, norm=None,
        residual=True, activation='leakyrelu', in_place_activation=True,
        transpose=False, reflectpad=True):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose
        self.reflectpad = reflectpad
        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)
        self.norm1 = None
        self.norm2 = None
        if norm is not None:
            if norm == 'batch':
                self.norm1 = nn.BatchNorm2d(out_channels)
                self.norm2 = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
                self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size=3, padding=0 if self.reflectpad else 1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                kernel_size=3, padding=0 if self.reflectpad else 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                padding=0 if self.reflectpad else 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=
                3, padding=0 if self.reflectpad else 1)
        if self.activation == 'relu':
            self.actfun1 = nn.ReLU(inplace=in_place_activation)
            self.actfun2 = nn.ReLU(inplace=in_place_activation)
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU(inplace=in_place_activation)
            self.actfun2 = nn.LeakyReLU(inplace=in_place_activation)
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU(inplace=in_place_activation)
            self.actfun2 = nn.ELU(inplace=in_place_activation)
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU(inplace=in_place_activation)
            self.actfun2 = nn.SELU(inplace=in_place_activation)
        if self.reflectpad:
            self.rpad1 = nn.ReflectionPad2d(1)
            self.rpad2 = nn.ReflectionPad2d(1)

    def forward(self, x):
        ox = x
        if self.reflectpad:
            x = self.rpad1(x)
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.actfun1(x)
        if self.norm1:
            x = self.norm1(x)
        if self.reflectpad:
            x = self.rpad2(x)
        x = self.conv2(x)
        if self.dropout:
            x = self.dropout2(x)
        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:, 0:min(ox.
                shape[1], x.shape[1]), :, :]
        x = self.actfun2(x)
        if self.norm2:
            x = self.norm2(x)
        return x


class Unet(nn.Module):

    def __init__(self, n_channel_in=1, n_channel_out=1, n_internal_channels
        =8, residual=True, down='avgpool', up='bilinear', activation=
        'leakyrelu', norm=None, softmax=False):
        super(Unet, self).__init__()
        self.residual = residual
        self.softmax = softmax
        nic = n_internal_channels
        if down == 'maxpool':
            self.down1 = nn.MaxPool2d(kernel_size=2)
            self.down2 = nn.MaxPool2d(kernel_size=2)
            self.down3 = nn.MaxPool2d(kernel_size=2)
            self.down4 = nn.MaxPool2d(kernel_size=2)
        elif down == 'avgpool':
            self.down1 = nn.AvgPool2d(kernel_size=2)
            self.down2 = nn.AvgPool2d(kernel_size=2)
            self.down3 = nn.AvgPool2d(kernel_size=2)
            self.down4 = nn.AvgPool2d(kernel_size=2)
        elif down == 'convpool':
            self.down1 = nn.Conv2d(nic, nic, kernel_size=2, stride=2, groups=32
                )
            self.down2 = nn.Conv2d(nic * 2, nic * 2, kernel_size=2, stride=
                2, groups=64)
            self.down3 = nn.Conv2d(nic * 4, nic * 4, kernel_size=2, stride=
                2, groups=128)
            self.down4 = nn.Conv2d(nic * 8, nic * 8, kernel_size=2, stride=
                2, groups=256)
            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
            self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
            self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25
            self.down1.bias.data = 0.01 * self.down1.bias.data + 0
            self.down2.bias.data = 0.01 * self.down2.bias.data + 0
            self.down3.bias.data = 0.01 * self.down3.bias.data + 0
            self.down4.bias.data = 0.01 * self.down4.bias.data + 0
        if up == 'bilinear' or up == 'nearest':
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up,
                scale_factor=2, align_corners=False)
            self.up2 = lambda x: nn.functional.interpolate(x, mode=up,
                scale_factor=2, align_corners=False)
            self.up3 = lambda x: nn.functional.interpolate(x, mode=up,
                scale_factor=2, align_corners=False)
            self.up4 = lambda x: nn.functional.interpolate(x, mode=up,
                scale_factor=2, align_corners=False)
        elif up == 'tconv':
            self.up1 = nn.ConvTranspose2d(nic * 8, nic * 8, kernel_size=2,
                stride=2, groups=nic * 8)
            self.up2 = nn.ConvTranspose2d(nic * 4, nic * 4, kernel_size=2,
                stride=2, groups=nic * 4)
            self.up3 = nn.ConvTranspose2d(nic * 2, nic * 2, kernel_size=2,
                stride=2, groups=nic * 2)
            self.up4 = nn.ConvTranspose2d(nic, nic, kernel_size=2, stride=2,
                groups=nic)
            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
            self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
            self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25
            self.up1.bias.data = 0.01 * self.up1.bias.data + 0
            self.up2.bias.data = 0.01 * self.up2.bias.data + 0
            self.up3.bias.data = 0.01 * self.up3.bias.data + 0
            self.up4.bias.data = 0.01 * self.up4.bias.data + 0
        self.conv1 = ConvBlock(n_channel_in, nic, residual=residual,
            activation=activation, norm=norm)
        self.conv2 = ConvBlock(nic, nic * 2, residual=residual, activation=
            activation, norm=norm)
        self.conv3 = ConvBlock(nic * 2, nic * 4, residual=residual,
            activation=activation, norm=norm)
        self.conv4 = ConvBlock(nic * 4, nic * 8, residual=residual,
            activation=activation, norm=norm)
        self.conv5 = ConvBlock(nic * 8, nic * 8, residual=residual,
            activation=activation, norm=norm)
        self.conv6 = ConvBlock(2 * nic * 8, nic * 4, residual=residual,
            activation=activation, norm=norm)
        self.conv7 = ConvBlock(2 * nic * 4, nic * 2, residual=residual,
            activation=activation, norm=norm)
        self.conv8 = ConvBlock(2 * nic * 2, nic, residual=residual,
            activation=activation, norm=norm)
        self.conv9 = ConvBlock(2 * nic, n_channel_out, residual=residual,
            activation=activation, norm=norm)
        if self.residual:
            self.convres = ConvBlock(n_channel_in, n_channel_out, residual=
                residual, activation=activation, norm=norm)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        if self.softmax:
            x = F.softmax(x, dim=1)
        x = self.conv5(x)
        x = self.up1(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        x = torch.cat([x, c4], 1)
        x = self.conv6(x)
        x = self.up2(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        x = torch.cat([x, c3], 1)
        x = self.conv7(x)
        x = self.up3(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        x = torch.cat([x, c2], 1)
        x = self.conv8(x)
        x = self.up4(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        x = torch.cat([x, c1], 1)
        x = self.conv9(x)
        if self.residual:
            x = torch.add(x, self.convres(c0))
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.pre11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=3, stride=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pre21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1)
        self.relu21 = nn.ReLU(inplace=True)
        self.pre22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1)
        self.relu22 = nn.ReLU(inplace=True)
        self.pre23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1)
        self.relu23 = nn.ReLU(inplace=True)
        self.pre24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=3, stride=1)
        self.relu24 = nn.ReLU(inplace=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pre31 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv31 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1)
        self.relu31 = nn.ReLU(inplace=True)
        self.pre32 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv32 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=3, stride=1)
        self.relu32 = nn.ReLU(inplace=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pre41 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv41 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1)
        self.relu41 = nn.ReLU(inplace=True)
        self.pre42 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv42 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size
            =3, stride=1)
        self.relu42 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pre11(x)
        x = self.relu11(self.conv11(x))
        x = self.up1(x)
        x = self.pre21(x)
        x = self.relu21(self.conv21(x))
        x = self.pre22(x)
        x = self.relu22(self.conv22(x))
        x = self.pre23(x)
        x = self.relu23(self.conv23(x))
        x = self.pre24(x)
        x = self.relu24(self.conv24(x))
        x = self.up2(x)
        x = self.pre31(x)
        x = self.relu31(self.conv31(x))
        x = self.pre32(x)
        x = self.relu32(self.conv32(x))
        x = self.up3(x)
        x = self.pre41(x)
        x = self.relu41(self.conv41(x))
        x = self.pre42(x)
        x = self.conv42(x)
        return x


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3,
            padding=0)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=
            3, padding=0)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel,
            kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(output_channel)
        self.upsample = upsample
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='bilinear', scale_factor=2)
        x_s = self.conv_shortcut(x)
        x = self.conv1(self.reflecPad1(x))
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(self.reflecPad2(x))
        x = self.relu(x)
        x = self.norm(x)
        return x_s + x


class Img_decoder_v3(nn.Module):

    def __init__(self):
        super(Img_decoder_v3, self).__init__()
        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = ResidualBlock(64, 64, upsample=False)
        self.map = nn.Conv2d(64, 5, 3, 1, 1)
        self.confidence = nn.Conv2d(64, 5, 3, 1, 1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, feat):
        h = self.slice4(feat)
        h = self.slice3(h)
        h = self.slice2(h)
        h = self.slice1(h)
        score = self.confidence(h)
        score = self.soft(score)
        out = self.map(h) * score
        out = torch.sum(out, dim=1).unsqueeze(1)
        return out


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


def concat(c1, c2):
    return torch.cat([c1, c2], dim=1)


def conv1x1(in_c, out_c, k, s):
    return nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s)


def conv3x3(in_c, out_c, k, s):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s)


def cut(c1, c2):
    x1, y1 = c1.size()[2:]
    x2, y2 = c2.size()[2:]
    c2 = c2[:, :, int((x2 - x1) / 2):int((x1 + x2) / 2), int((y2 - y1) / 2)
        :int((y2 + y1) / 2)]
    return c2


def upconv2x2(in_c, out_c, k, s):
    return nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s)


class UNET(nn.Module):

    def __init__(self, in_channels, n_classes):
        super(UNET, self).__init__()
        self.conv0 = conv3x3(in_channels, 64, 3, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(64, 64, 3, 1)
        self.maxpool_0 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv3x3(64, 128, 3, 1)
        self.conv3 = conv3x3(128, 128, 3, 1)
        self.conv4 = conv3x3(128, 256, 3, 1)
        self.conv5 = conv3x3(256, 256, 3, 1)
        self.conv6 = conv3x3(256, 512, 3, 1)
        self.conv7 = conv3x3(512, 512, 3, 1)
        self.conv8 = conv3x3(512, 1024, 3, 1)
        self.conv9 = conv3x3(1024, 1024, 3, 1)
        self.conv10 = conv3x3(1024, 512, 3, 1)
        self.conv11 = conv3x3(512, 256, 3, 1)
        self.conv12 = conv3x3(256, 128, 3, 1)
        self.conv13 = conv3x3(128, 64, 3, 1)
        self.conv14 = conv1x1(64, n_classes, 1, 1)
        self.upconv0 = upconv2x2(1024, 512, 2, 2)
        self.upconv1 = upconv2x2(512, 256, 2, 2)
        self.upconv2 = upconv2x2(256, 128, 2, 2)
        self.upconv3 = upconv2x2(128, 64, 2, 2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        stage1 = x
        x = self.maxpool_0(x)
        x = self.conv2(x)
        x = self.conv3(x)
        stage2 = x
        x = self.maxpool_0(x)
        x = self.conv4(x)
        x = self.conv5(x)
        stage3 = x
        x = self.maxpool_0(x)
        x = self.conv6(x)
        x = self.conv7(x)
        stage4 = x
        x = self.maxpool_0(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.upconv0(x)
        stage4 = cut(x, stage4)
        x = concat(x, stage4)
        x = self.conv10(x)
        x = self.conv7(x)
        x = self.upconv1(x)
        stage3 = cut(x, stage3)
        x = concat(x, stage3)
        x = self.conv11(x)
        x = self.conv5(x)
        x = self.upconv2(x)
        stage2 = cut(x, stage2)
        x = concat(x, stage2)
        x = self.conv12(x)
        x = self.conv3(x)
        x = self.upconv3(x)
        stage1 = cut(x, stage1)
        x = concat(x, stage1)
        x = self.conv13(x)
        x = self.conv1(x)
        x = self.conv14(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 256, 256])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'n_classes': 4}]

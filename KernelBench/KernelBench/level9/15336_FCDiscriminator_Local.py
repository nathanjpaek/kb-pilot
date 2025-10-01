import torch
import torch.nn as nn


class FCDiscriminator_Local(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator_Local, self).__init__()
        self.conv1 = nn.Conv2d(num_classes + 2048, ndf, kernel_size=4,
            stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1
            )
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2,
            padding=1)
        self.classifier = nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2,
            padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        return x


def get_inputs():
    return [torch.rand([4, 2052, 64, 64])]


def get_init_inputs():
    return [[], {'num_classes': 4}]

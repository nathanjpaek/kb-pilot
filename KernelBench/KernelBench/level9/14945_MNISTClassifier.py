import torch
import torchvision
import torchvision.ops
from torch import nn


class DeformableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size = kernel_size if type(kernel_size) == tuple else (
            kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] *
            kernel_size[1], kernel_size=kernel_size, stride=stride, padding
            =self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size[0] *
            kernel_size[1], kernel_size=kernel_size, stride=stride, padding
            =self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)
        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels
            =out_channels, kernel_size=kernel_size, stride=stride, padding=
            self.padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=
            self.regular_conv.weight, bias=self.regular_conv.bias, padding=
            self.padding, mask=modulator, stride=self.stride)
        return x


class MNISTClassifier(nn.Module):

    def __init__(self, deformable=False):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
            bias=True)
        conv = nn.Conv2d if deformable is False else DeformableConv2d
        self.conv4 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True
            )
        self.conv5 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True
            )
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]

import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class SamePad2dStrong(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2dStrong, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = (out_width - 1) * self.stride[0] + self.kernel_size[0
            ] - in_width
        pad_along_height = (out_height - 1) * self.stride[1
            ] + self.kernel_size[1] - in_height
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom),
            'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


class StrongMask(nn.Module):

    def __init__(self, depth=256, pool_size=14, num_classes=21):
        super(StrongMask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.padding = SamePad2dStrong(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.padding(x))
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 256, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
from collections import OrderedDict


class VGG19Decoder2(nn.Module):

    def __init__(self):
        super(VGG19Decoder2, self).__init__()
        self.blocks = OrderedDict([('pad2_1', nn.ReflectionPad2d(1)), (
            'conv2_1', nn.Conv2d(128, 64, 3, 1, 0)), ('relu2_1', nn.ReLU(
            inplace=True)), ('unpool1', nn.Upsample(scale_factor=2)), (
            'pad1_2', nn.ReflectionPad2d(1)), ('conv1_2', nn.Conv2d(64, 64,
            3, 1, 0)), ('relu1_2', nn.ReLU(inplace=True)), ('pad1_1', nn.
            ReflectionPad2d(1)), ('conv1_1', nn.Conv2d(64, 3, 3, 1, 0))])
        self.seq = nn.Sequential(self.blocks)

    def forward(self, x, targets=None):
        return self.seq(x)


def get_inputs():
    return [torch.rand([4, 128, 4, 4])]


def get_init_inputs():
    return [[], {}]

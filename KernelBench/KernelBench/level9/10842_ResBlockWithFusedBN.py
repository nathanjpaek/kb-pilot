import torch
from torch import nn
from torchvision import models as models
import torch.onnx


class ResBlockWithFusedBN(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation
        =1, group=1, stride_1x1=True):
        super().__init__()
        str1x1, str3x3 = (stride, 1) if stride_1x1 else (1, stride)
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride
            =str1x1, bias=True)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3,
            stride=str3x3, bias=True, padding=1 * dilation, dilation=
            dilation, groups=group)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1,
            stride=1, bias=True)
        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                stride=stride, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for submodule in self.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_uniform_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'outplanes': 4, 'innerplanes': 4}]

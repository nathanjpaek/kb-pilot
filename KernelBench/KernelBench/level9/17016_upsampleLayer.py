import torch
import torch.nn as nn


class upsampleLayer(nn.Module):
    """
    A upsample layer of UNet. ReLU is the activation func. The skip connection 
    can be cutted if not given. Because RGB-UV is not a completion task but a
    image transition task.
    """

    def __init__(self, infeature, outfeature, kernelSize, strides=1,
        paddings=1, bn=False, dropout_rate=0):
        super(upsampleLayer, self).__init__()
        self.upsp = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(infeature, outfeature, kernelSize, stride=
            strides, padding=paddings)
        self.acti = nn.ReLU()
        self.drop = None
        if dropout_rate != 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(outfeature, momentum=0.8)

    def forward(self, x, skip_input=None):
        y = self.conv(self.upsp(x))
        if self.drop is not None:
            y = self.drop(y)
        if self.bn is not None:
            y = self.bn(y)
        if skip_input is not None:
            y = torch.cat((y, skip_input), 1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'infeature': 4, 'outfeature': 4, 'kernelSize': 4}]

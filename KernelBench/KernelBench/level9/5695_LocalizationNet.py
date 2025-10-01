import torch
import torch.utils.data
import torch.nn as nn


class LocalizationNet(nn.Module):

    def __init__(self, inplanes, inputsize, nheads=1, use_bn=False):
        super(LocalizationNet, self).__init__()
        inputH, inputW = inputsize
        self.use_bn = use_bn
        if self.use_bn:
            None
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1
            )
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = None
        self.conv3 = None
        self.factor = 4
        self.channels = 64
        if inputH >= 8:
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.channels = 64
            self.factor = 8
            if self.use_bn:
                self.bn2 = nn.BatchNorm2d(64)
        if inputH >= 16:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.channels = 128
            self.factor = 16
            if self.use_bn:
                self.bn3 = nn.BatchNorm2d(128)
        self.nheads = nheads
        if self.nheads > 1:
            self.nlinear = []
            if self.nheads >= 5:
                self.conv3 = None
                self.factor = 8
                self.channels = 64
            fw = inputW // self.factor
            self.fw_strip = [(fw // nheads) for i in range(nheads)]
            if fw % nheads != 0:
                self.fw_strip[-1] += 1
            for i in range(nheads):
                self.nlinear.append(nn.Linear(self.channels * (inputH //
                    self.factor) * self.fw_strip[i], 30))
            self.nlinear = nn.ModuleList(self.nlinear)
        else:
            self.inplanes = self.channels * (inputH // self.factor) * (inputW
                 // self.factor)
            self.linear = nn.Linear(self.inplanes, 30)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.pool(x)
        if self.conv2:
            x = self.conv2(x)
            x = self.relu(x)
            if self.use_bn:
                x = self.bn2(x)
            x = self.pool(x)
        if self.conv3:
            x = self.conv3(x)
            x = self.relu(x)
            if self.use_bn:
                x = self.bn3(x)
            x = self.pool(x)
        if self.nheads > 1:
            b = x.size(0)
            tx = []
            start = 0
            for i in range(self.nheads):
                end = start + self.fw_strip[i]
                x_ = x[:, :, :, start:end]
                x_ = x_.reshape(b, -1)
                x_ = self.relu(self.nlinear[i](x_))
                start = end
                tx.append(x_)
            x = tx
        else:
            x = x.view(-1, self.inplanes)
            x = self.linear(x)
            x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'inputsize': [4, 4]}]

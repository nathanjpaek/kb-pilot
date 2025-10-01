import torch
import torch.nn as nn


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == 'batch':
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == 'instance':
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == 'layer':
        return nn.GroupNorm(1, n_filters)
    elif norm == 'act':
        return norms.ActNorm(n_filters, False)


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None,
        leak=0.2):
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1,
            bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=
            dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}]

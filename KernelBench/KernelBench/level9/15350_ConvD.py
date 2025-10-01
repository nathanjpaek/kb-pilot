import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):

    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()
        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)
        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = normalization(planes, norm)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = normalization(planes, norm)
        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


def get_inputs():
    return [torch.rand([4, 8, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]

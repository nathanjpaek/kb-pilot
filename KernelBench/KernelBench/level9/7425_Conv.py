import torch
import torch.utils.data
from torch import nn


class Conv(nn.Module):
    """
    2d卷积
    先batchnorm再ReLU，默认有ReLU但是没有BN
    默认小核
    """

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False,
        relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
            padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, '{} {}'.format(x.size()[1],
            self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp_dim': 4, 'out_dim': 4}]

import torch
import torch.nn as nn


class _Residual_Block(nn.Module):

    def __init__(self, inc=64, outc=64, groups=1):
        super(_Residual_Block, self).__init__()
        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc,
                kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
            self.conv_expand = None
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc,
            kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.InstanceNorm2d(outc, eps=0.001)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc,
            kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.InstanceNorm2d(outc, eps=0.001)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]

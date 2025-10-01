import torch
import torch.nn.functional as F
import torch.nn as nn


class CReLU_IN(nn.Module):

    def __init__(self, channels):
        super(CReLU_IN, self).__init__()
        self.bn = nn.InstanceNorm2d(channels * 2, eps=1e-05, momentum=0.1,
            affine=True)

    def forward(self, x):
        cat = torch.cat((x, -x), 1)
        x = self.bn(cat)
        return F.leaky_relu(x, 0.01, inplace=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]

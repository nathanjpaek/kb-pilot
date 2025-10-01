import torch
import torch.nn as nn
import torch.nn.functional as F


class ReduceBranch(nn.Module):

    def __init__(self, planes, stride=2):
        super(ReduceBranch, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out1 = self.conv1(self.avg_pool(x))
        shift_x = x[:, :, 1:, 1:]
        shift_x = F.pad(shift_x, (0, 1, 0, 1))
        out2 = self.conv2(self.avg_pool(shift_x))
        out = torch.cat([out1, out2], dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'planes': 4}]

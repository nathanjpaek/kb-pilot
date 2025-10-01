import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition_nobn(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition_nobn, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(x))
        out = F.avg_pool2d(out, 2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4}]

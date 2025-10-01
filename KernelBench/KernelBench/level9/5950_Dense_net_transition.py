import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense_net_transition(nn.Module):

    def __init__(self, nChannels, outChannels):
        super(Dense_net_transition, self).__init__()
        self.conv = nn.Conv2d(nChannels, outChannels, kernel_size=1, bias=False
            )

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, (2, 2), (2, 2), padding=0)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nChannels': 4, 'outChannels': 4}]

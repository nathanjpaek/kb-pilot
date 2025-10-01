import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class make_dilation_dense(nn.Module):

    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=
            kernel_size, padding=(kernel_size - 1) // 2 + 1, bias=True,
            dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nChannels': 4, 'growthRate': 4}]

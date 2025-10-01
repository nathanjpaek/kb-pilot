import torch
import torch.nn as nn


class Mlayer(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(Mlayer, self).__init__()
        m_s = torch.zeros([1, in_channel, 1, 1], requires_grad=True)
        self.m_s = torch.nn.Parameter(m_s)
        self.register_parameter('m_scale', self.m_s)
        self.func = nn.Identity()
        if in_channel != out_channel:
            self.func = nn.Conv2d(in_channels=in_channel, out_channels=
                out_channel, kernel_size=1, stride=stride, padding=0)

    def forward(self, input):
        x = input * self.m_s
        x = self.func(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]

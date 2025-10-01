from torch.nn import Module
import torch
from torch.nn import Conv2d
from torch.nn import Parameter
from torch.nn import Softmax
from torch.nn.parameter import Parameter


class PAM_Module(Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim
        out_channels = max(in_dim // 8, min(in_dim, 2))
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=
            out_channels, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=
            out_channels, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim,
            kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]

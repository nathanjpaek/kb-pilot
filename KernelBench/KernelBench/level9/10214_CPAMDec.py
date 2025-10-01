from torch.nn import Module
import torch
from torch.nn import Conv2d
from torch.nn import Parameter
from torch.nn import Softmax
from torch.nn import Linear
from torch.nn.parameter import Parameter


class CPAMDec(Module):
    """
    CPAM decoding module
    """

    def __init__(self, in_channels):
        super(CPAMDec, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))
        self.conv_query = Conv2d(in_channels=in_channels, out_channels=
            in_channels // 4, kernel_size=1)
        self.conv_key = Linear(in_channels, in_channels // 4)
        self.conv_value = Linear(in_channels, in_channels)

    def forward(self, x, y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize, C, width, height = x.size()
        m_batchsize, K, _M = y.size()
        proj_query = self.conv_query(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.conv_key(y).view(m_batchsize, K, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.conv_value(y).permute(0, 2, 1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.scale * out + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]

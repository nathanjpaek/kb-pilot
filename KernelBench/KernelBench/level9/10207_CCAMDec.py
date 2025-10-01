from torch.nn import Module
import torch
from torch.nn import Parameter
from torch.nn import Softmax
from torch.nn.parameter import Parameter


class CCAMDec(Module):
    """
    CCAM decoding module
    """

    def __init__(self):
        super(CCAMDec, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        m_batchsize, C, width, height = x.size()
        x_reshape = x.view(m_batchsize, C, -1)
        B, K, _W, _H = y.size()
        y_reshape = y.view(B, K, -1)
        proj_query = x_reshape
        proj_key = y_reshape.permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy
            ) - energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B, K, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, width, height)
        out = x + self.scale * out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

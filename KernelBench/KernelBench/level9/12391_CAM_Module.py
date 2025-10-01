from torch.nn import Module
import torch
from torch.nn import Parameter
from torch.nn import Softmax


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        Calcuate attetion between channels
        Args:
            x: input feature maps (B * C * H * W)

        Returns:
            out: attention value + input feature (B * C * H * W)
            attention: B * C * C

        """
        m_batchsize, C, height, wight = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_value = x.view(m_batchsize, C, -1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy
            ) - energy
        attention = self.softmax(energy_new)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, wight)
        mean = torch.mean(out)
        out = out / mean
        out = self.gamma * out + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]

import torch
import torch.nn as nn
import torch.utils.data


class CAM_Calculate(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Calculate, self).__init__()
        self.chanel_in = in_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                attention: B X C X C
        """
        m_batchsize, C, _height, _width = x.size()
        proj_query = x.contiguous().view(m_batchsize, C, -1)
        proj_key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy
            ) - energy
        attention = self.softmax(energy_new)
        return attention


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]

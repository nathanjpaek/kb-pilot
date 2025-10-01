import torch
import torch.nn as nn
import torch.utils.data


class CAM_Use(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Use, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, attention):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                attention: B X C X C
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        proj_value = x.contiguous().view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]

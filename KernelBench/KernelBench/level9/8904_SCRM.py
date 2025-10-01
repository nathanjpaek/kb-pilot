import torch
import torch.nn.functional as F
import torch.nn as nn


class SCRM(nn.Module):
    """
    spatial & channel wise relation loss
    """

    def __init__(self, gamma=0.1):
        super(SCRM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gamma

    def spatial_wise(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

    def channel_wise(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy
            ) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

    def cal_loss(self, f_s, f_t):
        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)
        sa_loss = F.l1_loss(self.spatial_wise(f_s), self.spatial_wise(f_t))
        ca_loss = F.l1_loss(self.channel_wise(f_s), self.channel_wise(f_t))
        return ca_loss + sa_loss

    def forward(self, g_s, g_t):
        return sum(self.cal_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

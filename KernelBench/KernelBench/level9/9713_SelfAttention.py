import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Linear(in_dim, in_dim)
        self.key_conv = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, num_dim = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, 1, num_dim).permute(
            0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, 1, num_dim)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, 1, num_dim)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, num_dim)
        out = self.gamma * out + x
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]

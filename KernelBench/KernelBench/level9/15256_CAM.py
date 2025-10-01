import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._utils


class CAM(nn.Module):

    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.para_mu = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, C, H, W = x.size()
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy, dim=-1)
        proj_value = x.view(N, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)
        out = self.para_mu * out + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]

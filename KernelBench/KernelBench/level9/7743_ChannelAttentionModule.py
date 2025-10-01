import torch
import torch.nn as nn
import torch.nn.functional as F


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    """forward"""

    def forward(self, x):
        return x * self.scale


class ChannelAttentionModule(nn.Module):

    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = Scale(scale=0)
    """forward"""

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy
            ) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)
        out = self.gamma(out) + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

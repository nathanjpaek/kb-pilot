import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed


class HighwayNetwork(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(HighwayNetwork, self).__init__()
        self.gate_proj = nn.Linear(in_dim, out_dim)
        self.lin_proj = nn.Linear(in_dim, out_dim)
        self.nonlin_proj = nn.Linear(in_dim, out_dim)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.constant_(p, 0)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x) - 2)
        lin = self.lin_proj(x)
        nonlin = torch.relu(self.nonlin_proj(x))
        res = gate * nonlin + (1 - gate) * lin
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]

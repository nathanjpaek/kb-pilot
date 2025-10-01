import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionInputDepth(nn.Module):

    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.convd = nn.Conv2d(64 + hidden_dim, out_chs - 1, 3, padding=1)

    def forward(self, depth, cost):
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))
        dfm = F.relu(self.convd1(depth))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cor, dfm], dim=1)
        out_d = F.relu(self.convd(cor_dfm))
        return torch.cat([out_d, depth], dim=1)


def get_inputs():
    return [torch.rand([4, 1, 64, 64]), torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'cost_dim': 4, 'hidden_dim': 4, 'out_chs': 4}]

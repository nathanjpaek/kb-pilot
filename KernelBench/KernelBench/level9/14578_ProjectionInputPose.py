import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionInputPose(nn.Module):

    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convp1 = nn.Conv2d(6, hidden_dim, 7, padding=3)
        self.convp2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.convp = nn.Conv2d(64 + hidden_dim, out_chs - 6, 3, padding=1)

    def forward(self, pose, cost):
        bs, _, h, w = cost.shape
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))
        pfm = F.relu(self.convp1(pose.view(bs, 6, 1, 1).repeat(1, 1, h, w)))
        pfm = F.relu(self.convp2(pfm))
        cor_pfm = torch.cat([cor, pfm], dim=1)
        out_p = F.relu(self.convp(cor_pfm))
        return torch.cat([out_p, pose.view(bs, 6, 1, 1).repeat(1, 1, h, w)],
            dim=1)


def get_inputs():
    return [torch.rand([4, 6, 1, 1]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cost_dim': 4, 'hidden_dim': 256, 'out_chs': 8}]

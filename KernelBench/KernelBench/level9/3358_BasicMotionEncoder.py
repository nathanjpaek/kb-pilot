from _paritybench_helpers import _mock_config
import torch
import torch.nn.functional as F
import torch.nn as nn


class BasicMotionEncoder(nn.Module):

    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


def get_inputs():
    return [torch.rand([4, 2, 64, 64]), torch.rand([4, 36, 64, 64])]


def get_init_inputs():
    return [[], {'args': _mock_config(corr_levels=4, corr_radius=4)}]

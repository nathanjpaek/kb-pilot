import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizeScale(nn.Module):

    def __init__(self, dim, init_norm=20):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        bottom_normalized = F.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled


class LocationEncoder(nn.Module):

    def __init__(self, init_norm, dim):
        super(LocationEncoder, self).__init__()
        self.lfeat_normalizer = NormalizeScale(5, init_norm)
        self.fc = nn.Linear(5, dim)

    def forward(self, lfeats):
        loc_feat = self.lfeat_normalizer(lfeats)
        output = self.fc(loc_feat)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 5])]


def get_init_inputs():
    return [[], {'init_norm': 4, 'dim': 4}]

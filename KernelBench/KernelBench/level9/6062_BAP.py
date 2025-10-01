import torch
import torch.nn as nn


class BAP(nn.Module):

    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B = features.size(0)
        M = attentions.size(1)
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B,
                1, -1)
            if i == 0:
                feature_matrix = AiF
            else:
                feature_matrix = torch.cat([feature_matrix, AiF], dim=1)
        return feature_matrix


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

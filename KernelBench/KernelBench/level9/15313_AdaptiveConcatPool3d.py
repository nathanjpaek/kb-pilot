import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConcatPool3d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool3d(x, 1), F.
            adaptive_max_pool3d(x, 1)), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn


class SameModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim + 1, 1, kernel_size=(1, 1))
        torch.nn.init.kaiming_normal_(self.conv.weight)
        self.dim = dim

    def forward(self, feats, attn):
        size = attn.size()[2]
        _the_max, the_idx = F.max_pool2d(attn, size, return_indices=True)
        attended_feats = feats.index_select(2, torch.div(the_idx[0, 0, 0, 0
            ], size, rounding_mode='floor'))
        attended_feats = attended_feats.index_select(3, the_idx[0, 0, 0, 0] %
            size)
        x = torch.mul(feats, attended_feats.repeat(1, 1, size, size))
        x = torch.cat([x, attn], dim=1)
        out = torch.sigmoid(self.conv(x))
        return out


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]

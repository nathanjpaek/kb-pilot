import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn


class QueryModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64]), torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'dim': 4}]

import torch
from torch import nn


class CatFusion(nn.Module):

    def __init__(self):
        super(CatFusion, self).__init__()

    def forward(self, seq_features, img_features, fuse_dim=1, **kwargs):
        return torch.cat((seq_features, img_features), dim=fuse_dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

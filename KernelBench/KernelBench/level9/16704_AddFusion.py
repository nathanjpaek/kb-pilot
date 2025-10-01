import torch
from torch import nn


class AddFusion(nn.Module):

    def __init__(self):
        super(AddFusion, self).__init__()

    def forward(self, seq_features, img_features, **kwargs):
        return seq_features + img_features


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

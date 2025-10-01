import torch
from torch import nn


class CatImgs(nn.Module):

    def forward(self, img1, img2, img3):
        return torch.cat((img1, img2, img3), 3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

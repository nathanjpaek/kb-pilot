import torch
from torch import nn
import torch.autograd


class LinfDistance(nn.Module):

    def forward(self, img1, img2):
        return (img1 - img2).reshape(img1.shape[0], -1).abs().max(dim=1)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

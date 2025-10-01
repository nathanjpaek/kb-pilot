import torch
from torch import nn
import torch.autograd


class L2Distance(nn.Module):

    def forward(self, img1, img2):
        return (img1 - img2).reshape(img1.shape[0], -1).norm(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

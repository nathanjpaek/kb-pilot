import torch
import torch.nn as nn
from torchvision import transforms as ttf


class ShearX(nn.Module):

    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 10 * self.M - 180

    def forward(self, img):
        return ttf.functional.affine(img, 0, [0, 0], 1, [self.angle, 0])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'M': 4}]

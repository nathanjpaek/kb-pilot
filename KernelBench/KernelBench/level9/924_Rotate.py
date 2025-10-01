import torch
import torch.nn as nn
from torchvision import transforms as ttf


class Rotate(nn.Module):

    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 10 * self.M

    def forward(self, img):
        return ttf.functional.rotate(img, self.angle)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'M': 4}]

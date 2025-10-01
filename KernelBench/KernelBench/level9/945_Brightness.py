import torch
import torch.nn as nn
from torchvision import transforms as ttf


class Brightness(nn.Module):

    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, img):
        return ttf.functional.adjust_brightness(img, self.M / 5.0)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'M': 4}]

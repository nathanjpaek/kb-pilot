import torch
import torch.nn as nn
from torchvision import transforms as ttf


class TranslateY(nn.Module):

    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, img):
        try:
            max_size = img.size()[1]
        except TypeError:
            max_size = img.size()[1]
        return ttf.functional.affine(img, 0, [0, (max_size - 1) / 10 * self
            .M], 1, [0, 0])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'M': 4}]

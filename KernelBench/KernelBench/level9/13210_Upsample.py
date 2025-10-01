import torch
import torch.nn as nn
import torch.nn.parallel


class Upsample(nn.Module):

    def __init__(self, n_iter):
        super(Upsample, self).__init__()
        self.n_iter = n_iter

    def forward(self, img):
        for _ in range(self.n_iter):
            img = nn.functional.interpolate(img, scale_factor=2.0, mode=
                'bicubic')
        return img


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_iter': 4}]

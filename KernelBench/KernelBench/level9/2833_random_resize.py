import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_4d_tensor_by_factor(x, height_factor, width_factor):
    res = F.interpolate(x, scale_factor=(height_factor, width_factor), mode
        ='bilinear')
    return res


class random_resize(nn.Module):

    def __init__(self, max_size_factor, min_size_factor):
        super().__init__()
        self.max_size_factor = max_size_factor
        self.min_size_factor = min_size_factor

    def forward(self, x):
        height_factor = random.uniform(a=self.min_size_factor, b=self.
            max_size_factor)
        width_factor = random.uniform(a=self.min_size_factor, b=self.
            max_size_factor)
        resized = resize_4d_tensor_by_factor(x=x, height_factor=
            height_factor, width_factor=width_factor)
        return resized


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_size_factor': 4, 'min_size_factor': 4}]

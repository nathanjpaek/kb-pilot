import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo


class Linear(nn.Module):

    def __init__(self, stride):
        super(Linear, self).__init__()
        self.scale = stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='linear')


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'stride': 1}]

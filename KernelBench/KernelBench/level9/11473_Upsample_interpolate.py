import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample_interpolate(nn.Module):

    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        x_numpy = x.cpu().detach().numpy()
        H = x_numpy.shape[2]
        W = x_numpy.shape[3]
        H = H * self.stride
        W = W * self.stride
        out = F.interpolate(x, size=(H, W), mode='nearest')
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'stride': 1}]

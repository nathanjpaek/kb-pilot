import torch
import torch.nn as nn


class Recover_from_density(nn.Module):

    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest'
            )

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        return torch.mul(x, out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'upscale_factor': 1.0}]

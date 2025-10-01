import torch
import torch.nn as nn


class UpsampleLayer(nn.Module):
    """

    """

    def __init__(self, scale_factor, mode='bilinear'):
        """

        :param scale_factor:
        :param mode:
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return nn.functional.interpolate(x, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0}]

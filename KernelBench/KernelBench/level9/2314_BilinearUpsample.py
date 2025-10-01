import torch
from typing import Union
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class BilinearUpsample(nn.Module):
    """
    Overview:
        Upsamples the input to the given member varible scale_factor using mode biliner
    Interface:
        forward
    """

    def __init__(self, scale_factor: 'Union[float, List[float]]') ->None:
        """
        Overview:
            Init class BilinearUpsample

        Arguments:
            - scale_factor (:obj:`Union[float, List[float]]`): multiplier for spatial size
        """
        super(BilinearUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Overview:
            Return the upsampled input
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - upsample(:obj:`torch.Tensor`): the upsampled input tensor
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode=
            'bilinear', align_corners=False)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0}]

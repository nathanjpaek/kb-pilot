import torch
from torchvision.datasets import *
import torch.nn as nn
from torchvision.transforms import *


class GramMatrix(nn.Module):
    """ Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \\mathcal{G} = \\sum_{h=1}^{H_i}\\sum_{w=1}^{W_i} \\mathcal{F}_{h,w}\\mathcal{F}_{h,w}^T
    """

    def forward(self, y):
        b, ch, h, w = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

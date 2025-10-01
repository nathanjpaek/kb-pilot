import torch
import torch.nn as nn


class DomainAdaptationLayer(nn.Module):
    """
        This class is for the Domain Adaptation Layer. For now, the layer works only in source domain

        arguments (function forward):
            image: the input image (type: tensor) (size: batch x 384 x W x H)
        return (function forward):
            image: the output image after concatenation (type: tensor)
    """

    def __init__(self):
        super(DomainAdaptationLayer, self).__init__()

    def forward(self, image):
        return torch.cat((image, image), 3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class NSGANLossGenerator(nn.Module):
    """
    This class implements the non-saturating generator GAN loss proposed in:
    https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
    """

    def __init__(self) ->None:
        """
        Constructor method.
        """
        super(NSGANLossGenerator, self).__init__()

    def forward(self, discriminator_prediction_fake: 'torch.Tensor', **kwargs
        ) ->torch.Tensor:
        """
        Forward pass.
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Non-saturating generator GAN loss
        """
        return F.softplus(-discriminator_prediction_fake).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

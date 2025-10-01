import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLossDiscriminator(nn.Module):
    """
    This class implements the standard discriminator GAN loss proposed in:
    https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
    """

    def __init__(self) ->None:
        """
        Constructor method.
        """
        super(GANLossDiscriminator, self).__init__()

    def forward(self, discriminator_prediction_real: 'torch.Tensor',
        discriminator_prediction_fake: 'torch.Tensor', **kwargs
        ) ->torch.Tensor:
        """
        Forward pass.
        :param discriminator_prediction_real: (torch.Tensor) Raw discriminator prediction for real samples
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Standard discriminator GAN loss
        """
        return F.softplus(-discriminator_prediction_real).mean() + F.softplus(
            discriminator_prediction_fake).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

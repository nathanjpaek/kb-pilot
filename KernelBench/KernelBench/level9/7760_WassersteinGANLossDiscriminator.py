import torch
import torch.nn as nn


class WassersteinGANLossDiscriminator(nn.Module):
    """
    This class implements the Wasserstein generator GAN loss proposed in:
    http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf
    """

    def __init__(self) ->None:
        """
        Constructor method.
        """
        super(WassersteinGANLossDiscriminator, self).__init__()

    def forward(self, discriminator_prediction_real: 'torch.Tensor',
        discriminator_prediction_fake: 'torch.Tensor', **kwargs
        ) ->torch.Tensor:
        """
        Forward pass.
        :param discriminator_prediction_real: (torch.Tensor) Raw discriminator prediction for real samples
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Wasserstein generator GAN loss with gradient penalty
        """
        return -discriminator_prediction_real.mean(
            ) + discriminator_prediction_fake.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

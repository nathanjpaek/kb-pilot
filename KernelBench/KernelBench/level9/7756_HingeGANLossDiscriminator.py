import torch
import torch.nn as nn


class HingeGANLossDiscriminator(nn.Module):
    """
    This class implements the Hinge discriminator GAN loss proposed in:
    https://arxiv.org/pdf/1705.02894.pdf
    """

    def __init__(self) ->None:
        """
        Constructor method.
        """
        super(HingeGANLossDiscriminator, self).__init__()

    def forward(self, discriminator_prediction_real: 'torch.Tensor',
        discriminator_prediction_fake: 'torch.Tensor', **kwargs
        ) ->torch.Tensor:
        """
        Forward pass.
        :param discriminator_prediction_real: (torch.Tensor) Raw discriminator prediction for real samples
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Hinge discriminator GAN loss
        """
        return -torch.minimum(torch.tensor(0.0, dtype=torch.float, device=
            discriminator_prediction_real.device), 
            discriminator_prediction_real - 1.0).mean() - torch.minimum(torch
            .tensor(0.0, dtype=torch.float, device=
            discriminator_prediction_fake.device), -
            discriminator_prediction_fake - 1.0).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

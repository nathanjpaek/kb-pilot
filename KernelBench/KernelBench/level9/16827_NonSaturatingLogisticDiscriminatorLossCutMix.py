import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F


class NonSaturatingLogisticDiscriminatorLossCutMix(nn.Module):
    """
    Implementation of the non saturating GAN loss for the discriminator network when performing cut mix augmentation.
    """

    def __init__(self) ->None:
        """
        Constructor
        """
        super(NonSaturatingLogisticDiscriminatorLossCutMix, self).__init__()

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction: (torch.Tensor)
        :return: (Tuple[torch.Tensor, torch.Tensor]) Loss values for real and fake part
        """
        loss_real = torch.mean(F.softplus(-prediction) * label)
        loss_fake = torch.mean(F.softplus(prediction) * (-label + 1.0))
        return loss_real, loss_fake


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

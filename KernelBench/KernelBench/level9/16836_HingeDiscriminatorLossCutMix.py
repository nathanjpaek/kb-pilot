import torch
from typing import Tuple
import torch.nn as nn


class HingeDiscriminatorLossCutMix(nn.Module):
    """
    This class implements the hinge gan loss for the discriminator network when utilizing cut mix augmentation.
    """

    def __init__(self) ->None:
        """
        Constructor method
        """
        super(HingeDiscriminatorLossCutMix, self).__init__()

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction: (torch.Tensor)
        :return: (Tuple[torch.Tensor, torch.Tensor]) Loss values for real and fake part
        """
        loss_real = -torch.mean(torch.minimum(torch.tensor(0.0, dtype=torch
            .float, device=prediction.device), prediction - 1.0) * label)
        loss_fake = -torch.mean(torch.minimum(torch.tensor(0.0, dtype=torch
            .float, device=prediction.device), -prediction - 1.0) * (-label +
            1.0))
        return loss_real, loss_fake


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

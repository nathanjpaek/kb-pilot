import torch
from torch import Tensor
from typing import NoReturn
from torch import nn


class BCEWithLogitsWithClassWeightLoss(nn.BCEWithLogitsLoss):
    """ finished, checked,
    """
    __name__ = 'BCEWithLogitsWithClassWeightsLoss'

    def __init__(self, class_weight: 'Tensor') ->NoReturn:
        """ finished, checked,

        Parameters
        ----------
        class_weight: Tensor,
            class weight, of shape (1, n_classes)
        """
        super().__init__(reduction='none')
        self.class_weight = class_weight

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            the prediction tensor, of shape (batch_size, ...)
        target: Tensor,
            the target tensor, of shape (batch_size, ...)

        Returns
        -------
        loss: Tensor,
            the loss (scalar tensor) w.r.t. `input` and `target`
        """
        loss = super().forward(input, target)
        loss = torch.mean(loss * self.class_weight)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'class_weight': 4}]

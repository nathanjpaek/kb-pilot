import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    This class implements the dice loss for multiple instances
    """

    def __init__(self, smooth_factor: 'float'=1.0) ->None:
        super(DiceLoss, self).__init__()
        self.smooth_factor = smooth_factor

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}, smooth factor={}'.format(self.__class__.__name__, self.
            smooth_factor)

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        loss = torch.tensor(1.0, dtype=torch.float32, device=prediction.device
            ) - (2.0 * torch.sum(torch.mul(prediction, label)) + self.
            smooth_factor) / (torch.sum(prediction) + torch.sum(label) +
            self.smooth_factor)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

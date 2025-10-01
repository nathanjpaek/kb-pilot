import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    Implementation of the dice loss proposed in:
    https://arxiv.org/abs/1707.03237
    """

    def __init__(self, smooth: 'float'=1.0) ->None:
        """
        Constructor method
        :param smooth: (float) Smoothness factor used in computing the dice loss
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward method calculates the dice loss
        :param prediction: (torch.tensor) Prediction tensor including probabilities
        :param label: (torch.tensor) Label tensor (one-hot encoded)
        :return: (torch.tensor) Dice loss
        """
        prediction = prediction.view(-1)
        label = label.view(-1)
        intersect = torch.sum(prediction * label) + self.smooth
        union = torch.sum(prediction) + torch.sum(label) + self.smooth
        dice_loss = 1.0 - 2.0 * intersect / union
        return dice_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

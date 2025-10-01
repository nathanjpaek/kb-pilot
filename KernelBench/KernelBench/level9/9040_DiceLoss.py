import torch
from torch import nn


class DiceLoss(nn.Module):
    """ Loss function based on Dice-Sorensen Coefficient (L = 1 - Dice)
  Input arguments:
    soft : boolean, default = True
           Select whether to use soft labelling or not. If true, dice calculated
           directly on sigmoid output without converting to binary. If false,
           sigmoid output converted to binary based on threshold value
    smooth : float, default = 1e-7
             Smoothing value to add to numerator and denominator of Dice. Low
             value will prevent inf or nan occurrence.
    threshold : float, default = 0.5
                Threshold of sigmoid activation values to convert to binary.
                Only applied if soft=False.
  """

    def __init__(self, soft=True, threshold=0.5, eps=1e-07):
        super().__init__()
        self.eps = eps
        self.soft = soft
        self.threshold = threshold

    def forward(self, inputs, targets):
        if not self.soft:
            inputs = BinaryDice(inputs, self.threshold)
        inputs = inputs.view(-1).float()
        targets = targets.view(-1).float()
        intersection = torch.sum(inputs * targets)
        dice = (2.0 * intersection + self.eps) / (torch.sum(inputs) + torch
            .sum(targets) + self.eps)
        return 1 - dice

    @staticmethod
    def BinaryDice(image, threshold=0.5):
        return (image > threshold).int()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

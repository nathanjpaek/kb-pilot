import torch
import torch.nn as nn


def flatten_channels(inputs, targets, channel_dim):
    """
    Helper function to flatten inputs and targets for each channel

    E.g., (1, 3, 10, 256, 256) --> (3, 655360)

    Parameters
    ----------
    inputs: torch.Tensor
        U-net output
    targets: torch.Tensor
        Target labels
    channel_dim: int
        Which dim represents output channels? 
    """
    order = [channel_dim]
    for i in range(len(inputs.shape)):
        if i != channel_dim:
            order.append(i)
    inputs = inputs.permute(*order)
    inputs = torch.flatten(inputs, start_dim=1)
    targets = targets.permute(*order)
    targets = torch.flatten(targets, start_dim=1)
    return inputs, targets


class DiceLoss(nn.Module):
    """
    DiceLoss: 1 - DICE coefficient 

    Adaptations: weights output channels equally in final loss. 
    This is necessary for anisotropic data.
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, channel_dim=1, smooth=1):
        """
        inputs: torch.tensor
            Network predictions. Float
        targets: torch.tensor
            Ground truth labels. Float
        channel_dim: int
            Dimension in which output channels can be found.
            Loss is weighted equally between output channels.
        smooth: int
            Smoothing hyperparameter.
        """
        inputs, targets = flatten_channels(inputs, targets, channel_dim)
        intersection = (inputs * targets).sum(-1)
        dice = (2.0 * intersection + smooth) / (inputs.sum(-1) + targets.
            sum(-1) + smooth)
        loss = 1 - dice
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

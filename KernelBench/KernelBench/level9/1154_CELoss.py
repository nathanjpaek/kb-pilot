import torch
import torch.nn as nn


class CELoss(nn.Module):
    """ Cross Entorpy Loss Wrapper

    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        """Forward function."""
        return self.loss_weight * self.criterion(output, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

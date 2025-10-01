import torch
import torch.nn as nn


class DiscShiftLoss(nn.Module):
    """Disc shift loss.

        Args:
            loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight=0.1):
        super(DiscShiftLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (n, c, h, w)

        Returns:
            Tensor: Loss.
        """
        loss = torch.mean(x ** 2)
        return loss * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

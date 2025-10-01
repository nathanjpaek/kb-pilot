import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """FocalLoss.

    .. seealso::
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        Proceedings of the IEEE international conference on computer vision. 2017.

    Args:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.

    Attributes:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.
    """

    def __init__(self, gamma=2, alpha=0.25, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1.0 - self.eps)
        cross_entropy = -(target * torch.log(input) + (1 - target) * torch.
            log(1 - input))
        logpt = -cross_entropy
        pt = torch.exp(logpt)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = -at * logpt
        focal_loss = balanced_cross_entropy * (1 - pt) ** self.gamma
        return focal_loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

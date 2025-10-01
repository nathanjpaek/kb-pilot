import torch
import torch.nn.functional as F
from torch import nn as nn


class FocalLoss(nn.Module):
    """Focal loss function for imbalanced dataset. 
    Args:
        alpha (float): weighing factor between 0 and 1. Alpha may be set by inverse 
                       class frequency
        gamma (float):  modulating factor reduces the loss contribution from easy 
                        examples and extends the range in which an example receives 
                        low loss. Usually between 0 - 5.  
    """

    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, y):
        bce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction=
            'none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

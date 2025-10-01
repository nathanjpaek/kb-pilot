import torch
import numpy as np
import torch.nn as nn


class BCEWithLogitsLoss2d(nn.Module):
    """Computationally stable version of 2D BCE loss

    """

    def __init__(self, weight=None, reduction='elementwise_mean'):
        super(BCEWithLogitsLoss2d, self).__init__()
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight)
        self.bce_loss = nn.BCEWithLogitsLoss(weight, reduction)

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(logits_flat, targets_flat)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

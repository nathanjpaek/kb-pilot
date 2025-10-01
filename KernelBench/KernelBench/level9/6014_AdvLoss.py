import torch
import torch.nn as nn


class AdvLoss(nn.Module):
    """BCE for True and False reals"""

    def __init__(self, alpha=1):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.loss_fn(pred, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

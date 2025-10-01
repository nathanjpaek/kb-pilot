import torch
import torch.nn.functional as F
import torch.nn as nn


class LSoftLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, beta):
        with torch.no_grad():
            y_true_updated = beta * y_true + (1 - beta) * y_pred
        return F.binary_cross_entropy(y_pred, y_true_updated, reduction='none')


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

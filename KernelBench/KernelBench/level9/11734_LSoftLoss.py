import torch
from torch import nn
import torch.nn.functional as F


def l_soft(y_pred, y_true, beta):
    eps = 1e-07
    y_pred = torch.clamp(y_pred, eps, 1.0)
    with torch.no_grad():
        y_true_update = beta * y_true + (1 - beta) * y_pred
    loss = F.binary_cross_entropy(y_pred, y_true_update)
    return loss


class LSoftLoss(nn.Module):

    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        output = torch.sigmoid(output)
        return l_soft(output, target, self.beta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

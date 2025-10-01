import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product


class FocalLossSigmoid(nn.Module):
    """
    sigmoid version focal loss
    """

    def __init__(self, alpha=0.25, gamma=2, size_average=False):
        super(FocalLossSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        inputs.size(0)
        inputs.size(1)
        P = torch.sigmoid(inputs)
        alpha_mask = self.alpha * targets
        loss_pos = -1.0 * torch.pow(1 - P, self.gamma) * torch.log(P
            ) * targets * alpha_mask
        loss_neg = -1.0 * torch.pow(1 - P, self.gamma) * torch.log(1 - P) * (
            1 - targets) * (1 - alpha_mask)
        batch_loss = loss_neg + loss_pos
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

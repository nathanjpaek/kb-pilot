import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = (logits and F.binary_cross_entropy_with_logits or F.
            binary_cross_entropy)
        self.reduce = reduce

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * torch.pow(1 - pt, self.gamma)
        focal_loss = focal_weight * bce
        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

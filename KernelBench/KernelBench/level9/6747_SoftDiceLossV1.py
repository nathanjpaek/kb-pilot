import torch
import torch.nn as nn


class SoftDiceLossV1(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1, reduction='mean'):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        args: logits: tensor of shape (N, H, W)
        args: label: tensor of shape(N, H, W)
        """
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum(dim=(1, 2))
        denor = (probs.pow(self.p) + labels).sum(dim=(1, 2))
        loss = 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

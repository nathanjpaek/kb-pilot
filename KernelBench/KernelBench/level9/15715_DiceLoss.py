import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        if torch.sum(target) == 0:
            output = 1.0 - output
            target = 1.0 - target
        return 1.0 - (2 * torch.sum(output * target) + self.smooth) / (
            torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
from torch import nn
import torch.backends.cudnn


class DiceLoss(nn.Module):

    def __init__(self, smooth=0, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (torch.
            sum(output) + torch.sum(target) + self.smooth + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

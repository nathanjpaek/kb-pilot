import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [(t.flatten(1).sum(1) if self.image else t.sum()) for t in [
            x * y, x + y]]
        dc = (2 * i + 1) / (u + 1)
        dc = 1 - dc.mean()
        return dc


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


class JaccardLoss(nn.Module):

    def __init__(self, apply_softmax: 'bool'=False, eps: 'float'=1e-06):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.eps = eps

    def forward(self, x, y, eps=1e-06):
        if self.apply_softmax:
            x = torch.softmax(x, dim=1)
        x = x.view(-1)
        y = y.reshape(-1)
        intersection = (x * y).sum()
        total = (x + y).sum()
        union = total - intersection
        IoU = (intersection + eps) / (union + eps)
        return 1 - IoU


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

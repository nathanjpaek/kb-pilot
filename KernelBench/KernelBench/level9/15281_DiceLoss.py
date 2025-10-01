import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, eps=1e-06):
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

    def forward(self, pred, target, mask=None):
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        if mask is not None:
            mask = mask.contiguous().view(mask.size()[0], -1)
            pred = pred * mask
            target = target * mask
        a = torch.sum(pred * target)
        b = torch.sum(pred)
        c = torch.sum(target)
        d = 2 * a / (b + c + self.eps)
        return 1 - d


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

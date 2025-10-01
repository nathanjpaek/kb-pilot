import torch
import torch.nn as nn


class EpeLossWithMask(nn.Module):

    def __init__(self, eps=1e-08, q=None):
        super(EpeLossWithMask, self).__init__()
        self.eps = eps
        self.q = q

    def forward(self, pred, label, mask):
        if self.q is not None:
            loss = ((pred - label).abs().sum(1) + self.eps) ** self.q
        else:
            loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        loss = loss * mask.squeeze(1)
        loss = loss.view(loss.shape[0], -1).sum(1) / mask.view(mask.shape[0
            ], -1).sum(1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

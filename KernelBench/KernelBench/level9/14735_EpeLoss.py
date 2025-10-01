import torch
import torch.nn as nn


class EpeLoss(nn.Module):

    def __init__(self, eps=0):
        super(EpeLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, label):
        loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        return loss.view(loss.shape[0], -1).mean(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

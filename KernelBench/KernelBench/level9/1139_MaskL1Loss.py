import torch
from torch import nn


class MaskL1Loss(nn.Module):

    def __init__(self, eps=1e-06):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: 'torch.Tensor', gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

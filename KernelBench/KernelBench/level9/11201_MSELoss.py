import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self, ratio=1, size_average=None, reduce=None, reduction=
        'mean'):
        super(MSELoss, self).__init__()
        self.ratio = ratio
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target, avg_factor=None):
        return self.ratio * F.mse_loss(input, target, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

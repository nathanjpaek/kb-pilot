import torch
from torch import nn
import torch.nn.functional as F


class MixedCycleLoss(nn.Module):

    def __init__(self, reduction: 'str'='none') ->None:
        super(MixedCycleLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input_2d, input_3d, target_2d, target_3d, w_cycle=1,
        w_3d=1):
        loss_cycle = F.mse_loss(input_2d, target_2d, reduction=self.reduction)
        loss_3d = F.mse_loss(input_3d, target_3d, reduction=self.reduction)
        mixed_loss = w_cycle * loss_cycle + w_3d * loss_3d
        return mixed_loss, loss_cycle, loss_3d


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

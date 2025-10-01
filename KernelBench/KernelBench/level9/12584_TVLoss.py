import torch
import torch.nn as nn


class TVLoss(nn.Module):

    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, out, gt):
        grad_out_x = out[:, :, :, 1:] - out[:, :, :, :-1]
        grad_out_y = out[:, :, 1:, :] - out[:, :, :-1, :]
        grad_gt_x = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        grad_gt_y = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        loss_x = self.l1(grad_out_x, grad_gt_x)
        loss_y = self.l1(grad_out_y, grad_gt_y)
        loss = self.weight * (loss_x + loss_y)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

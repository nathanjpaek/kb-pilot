import torch
import torch.nn as nn
import torch.utils.data


class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=0.001):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target, weight=None):
        pred_ctr_2x = pred[:, :2] + pred[:, 2:]
        pred_wh = pred[:, 2:] - pred[:, :2]
        with torch.no_grad():
            target_ctr_2x = target[:, :2] + target[:, 2:]
            target_wh = target[:, 2:] - target[:, :2]
        d_xy_2x = (target_ctr_2x - pred_ctr_2x).abs()
        loss_xy = torch.clamp((target_wh - d_xy_2x) / (target_wh + d_xy_2x +
            self.eps), min=0)
        loss_wh = torch.min(target_wh / (pred_wh + self.eps), pred_wh / (
            target_wh + self.eps))
        loss = 1 - torch.cat([loss_xy, loss_wh], dim=-1)
        if self.beta >= 1e-05:
            loss = torch.where(loss < self.beta, 0.5 * loss ** 2 / self.
                beta, loss - 0.5 * self.beta)
        if weight is not None:
            loss = loss * weight
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

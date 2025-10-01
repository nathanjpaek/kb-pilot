import torch
import torch.nn as nn


def cd_loss(preds, gts):

    def batch_pairwise_dist(x, y):
        _bs, num_points_x, _points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.
            transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P
    P = batch_pairwise_dist(gts, preds)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.sum(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.sum(mins)
    return loss_1 + loss_2


class ChamferLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, preds, gts, **kwargs):
        return cd_loss(preds, gts)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]

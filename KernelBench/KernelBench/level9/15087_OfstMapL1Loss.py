import torch
import torch.nn as nn


class OfstMapL1Loss(nn.Module):

    def __init__(self, eps=1e-05):
        super().__init__()
        self.eps = eps

    def forward(self, rgb_labels, pred, gt, normalize=True, reduce=True):
        wgt = (rgb_labels > 1e-08).float()
        bs, n_kpts, c, h, w = pred.size()
        wgt = wgt.view(bs, 1, 1, h, w).repeat(1, n_kpts, c, 1, 1).contiguous()
        diff = pred - gt
        abs_diff = torch.abs(diff)
        abs_diff = wgt * abs_diff
        in_loss = abs_diff
        if normalize:
            in_loss = torch.sum(in_loss.view(bs, n_kpts, -1), 2) / (torch.
                sum(wgt.view(bs, n_kpts, -1), 2) + 0.001)
        if reduce:
            in_loss = torch.mean(in_loss)
        return in_loss


def get_inputs():
    return [torch.rand([4, 1, 1, 4, 4]), torch.rand([4, 4, 4, 4, 4]), torch
        .rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

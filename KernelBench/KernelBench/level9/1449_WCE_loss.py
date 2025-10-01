import torch
import torch.nn as nn


class WCE_loss(nn.Module):

    def __init__(self):
        super(WCE_loss, self).__init__()

    def sum_ij(self, x):
        return torch.sum(torch.sum(x, dim=3), dim=2)

    def forward(self, pred, gt):
        N_fg = self.sum_ij(gt)
        N_bg = self.sum_ij(1 - gt)
        L_fg = -1 * self.sum_ij(torch.log(pred + 1e-16) * gt) / N_fg
        L_bg = -1 * self.sum_ij(torch.log(1 - pred + 1e-16) * (1 - gt)) / N_bg
        L = L_fg + L_bg
        return torch.mean(L)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

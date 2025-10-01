import torch
from torch import nn
from torch.optim.lr_scheduler import *


class BridgeFeatLoss(nn.Module):

    def __init__(self):
        super(BridgeFeatLoss, self).__init__()

    def forward(self, feats_s, feats_t, feats_mixed, lam):
        dist_mixed2s = ((feats_mixed - feats_s) ** 2).sum(1, keepdim=True)
        dist_mixed2t = ((feats_mixed - feats_t) ** 2).sum(1, keepdim=True)
        dist_mixed2s = dist_mixed2s.clamp(min=1e-12).sqrt()
        dist_mixed2t = dist_mixed2t.clamp(min=1e-12).sqrt()
        dist_mixed = torch.cat((dist_mixed2s, dist_mixed2t), 1)
        lam_dist_mixed = (lam * dist_mixed).sum(1, keepdim=True)
        loss = lam_dist_mixed.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 2, 4, 4])]


def get_init_inputs():
    return [[], {}]

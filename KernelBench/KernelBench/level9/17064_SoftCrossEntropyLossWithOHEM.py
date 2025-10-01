import torch
from torch import nn
import torch.nn.functional as F


def _ohem_mask(loss, ohem_ratio):
    with torch.no_grad():
        values, _ = torch.topk(loss.reshape(-1), int(loss.nelement() *
            ohem_ratio))
        mask = loss >= values[-1]
    return mask.float()


class SoftCrossEntropyLossWithOHEM(nn.Module):

    def __init__(self, ohem_ratio=1.0, eps=1e-07):
        super(SoftCrossEntropyLossWithOHEM, self).__init__()
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        loss = -(pred * target).sum(1)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def set_ohem_ratio(self, ohem_ratio):
        self.ohem_ratio = ohem_ratio


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn.functional as F
from torch import nn


class OHEMLoss(nn.Module):

    def __init__(self, rate=0.8):
        super(OHEMLoss, self).__init__()
        None
        self.rate = rate

    def change_rate(self, new_rate):
        None
        self.rate = new_rate

    def forward(self, cls_pred, cls_target):
        batch_size = cls_pred.size(0)
        ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction=
            'none', ignore_index=-1)
        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * self.rate))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocUnetLoss_DL_batch(nn.Module):
    """
    只使用一个unet的loss 目前使用这个loss训练的比较好
    """

    def __init__(self, r=0.0, reduction='mean'):
        super(DocUnetLoss_DL_batch, self).__init__()
        assert reduction in ['mean', 'sum'
            ], " reduction must in ['mean','sum']"
        self.r = r
        self.reduction = reduction

    def forward(self, y, label):
        _bs, _n, _h, _w = y.size()
        d = y - label
        loss1 = []
        for d_i in d:
            loss1.append(torch.abs(d_i).mean() - self.r * torch.abs(d_i.mean())
                )
        loss1 = torch.stack(loss1)
        loss2 = F.mse_loss(y, label, reduction=self.reduction)
        if self.reduction == 'mean':
            loss1 = loss1.mean()
        elif self.reduction == 'sum':
            loss1 = loss1.sum()
        return loss1 + loss2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

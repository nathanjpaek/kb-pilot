import torch
import torch.utils.data.distributed
import torch
import torch.nn as nn
from numpy import int64 as int64
import torch.utils


class SigmoidFocalLoss(nn.Module):

    def __init__(self, ignore_label, gamma=2.0, alpha=0.25, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, _h, _w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = target.ne(self.ignore_label).float()
        target = mask * target
        onehot = target.view(b, -1, 1)
        max_val = (-pred_sigmoid).clamp(min=0)
        pos_part = (1 - pred_sigmoid) ** self.gamma * (pred_sigmoid - 
            pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + ((-max_val).exp(
            ) + (-pred_sigmoid - max_val).exp()).log())
        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(dim
            =-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 16]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'ignore_label': 4}]

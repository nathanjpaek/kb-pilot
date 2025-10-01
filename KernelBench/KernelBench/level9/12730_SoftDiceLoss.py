import torch
import numpy as np
import torch.nn as nn


class IoU(nn.Module):

    def __init__(self, mode='iou', axis=1, eps=0.0):
        """ Return a matrix of [batch * num_classes]. 
            Note: In order to separate from iou=0, function WILL return NaN if both 
            y_true and y_pred are 0. Need further treatment to remove nan in either 
            loss function or matrix.
        """
        super(IoU, self).__init__()
        assert mode in ['iou', 'dice']
        self.factor = {'iou': -1.0, 'dice': 0.0}[mode]
        self.eps = eps
        self.axis = axis

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        sum_axis = list(range(1, self.axis)) + list(range(self.axis + 1,
            y_pred.ndim))
        prod = (y_true * y_pred).sum(sum_axis)
        plus = (y_true + y_pred).sum(sum_axis)
        iou = (2 + self.factor) * prod / (plus + self.factor * prod + self.eps)
        return iou


class SoftDiceLoss(IoU):

    def __init__(self, weight=None, ignore_index=[], reduction='mean', mode
        ='dice', axis=1, eps=0.0):
        super(SoftDiceLoss, self).__init__(mode, axis, eps)
        self.ignore_index = ignore_index
        self.register_buffer('weight', weight)
        self.reduction = {'none': lambda x: x, 'mean': torch.mean, 'sum':
            torch.sum}[reduction]

    def _apply_weight(self, x):
        """ Apply class_weights to calculate loss, ignore nan. """
        if self.weight is None:
            weight = torch.ones(x.shape[-1], device=x.device)
        else:
            weight = self.weight
        idx = np.ones(x.shape[-1], dtype=bool)
        idx[self.ignore_index] = False
        x, weight = x[:, idx], weight[idx]
        weight = ~torch.isnan(x) * weight
        return x * weight / weight.sum(-1, keepdim=True)

    def forward(self, y_pred, y_true):
        iou = super(SoftDiceLoss, self).forward(y_pred, y_true)
        iou = self._apply_weight(iou)
        return -self.reduction(iou.sum(-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

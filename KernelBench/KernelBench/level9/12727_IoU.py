import torch
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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

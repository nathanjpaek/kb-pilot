import torch
import torch.nn as nn


class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, weight=None, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)

    def forward(self, pred, target):
        return super(SegmentationLosses, self).forward(pred, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

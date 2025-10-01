import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed


class IoULoss(nn.Module):
    """
    Creates a criterion that computes the Intersection over Union (IoU)
    between a segmentation mask and its ground truth.

    Rahman, M.A. and Wang, Y:
    Optimizing Intersection-Over-Union in Deep Neural Networks for
    Image Segmentation. International Symposium on Visual Computing (2016)
    http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        input = F.sigmoid(input)
        intersection = (input * target).sum()
        union = (input + target - input * target).sum()
        iou = intersection / union
        iou_dual = input.size(0) - iou
        if self.size_average:
            iou_dual = iou_dual / input.size(0)
        return iou_dual


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

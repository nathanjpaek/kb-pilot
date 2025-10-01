import torch
from torch.nn.modules.loss import _Loss


class SoftDiceLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, y_pred, y_gt):
        numerator = torch.sum(y_pred * y_gt)
        denominator = torch.sum(y_pred * y_pred + y_gt * y_gt)
        return numerator / denominator


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

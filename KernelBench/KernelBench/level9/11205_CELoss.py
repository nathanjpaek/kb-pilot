import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


class CELoss(nn.Module):

    def __init__(self, ratio=1, weight=None, size_average=None,
        ignore_index=-100, reduce=None, reduction='mean'):
        super(CELoss, self).__init__()
        self.ratio = ratio
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        Calculate the cross-entropy loss
        :param input(torch.Tensor): The prediction with shape (N, C),
                                    C is the number of classes.
        :param target(torch.Tensor): The learning label(N, 1) of
                                     the prediction.
        :return: (torch.Tensor): The calculated loss
        """
        target = target.squeeze_()
        return self.ratio * F.cross_entropy(input, target, weight=self.
            weight, ignore_index=self.ignore_index, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch._C
import torch.serialization
from torch import nn
import torch.nn.functional as F
from typing import *


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class F1Loss(nn.Module):
    """F1Loss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', rp_weight=[1.0, 1.0], class_weight
        =None, loss_weight=1.0):
        super(F1Loss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.rp_weight = rp_weight
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.smooth = 1e-06

    def forward(self, predict, target, weight=None, avg_factor=None,
        reduction_override=None, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.
            reduction)
        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight).type_as(predict)
        else:
            class_weight = None
        N, _C, H, W = predict.size()
        _, maxpred = torch.max(predict, 1)
        predict_onehot = torch.zeros(predict.size()).type_as(maxpred)
        predict_onehot.scatter_(1, maxpred.view(N, 1, H, W), 1)
        target_onehot = torch.zeros(predict.size()).type_as(target)
        target_onehot.scatter_(1, target.view(N, 1, H, W), 1)
        true_positive = torch.sum(predict_onehot * target_onehot, dim=(2, 3))
        total_target = torch.sum(target_onehot, dim=(2, 3))
        total_predict = torch.sum(predict_onehot, dim=(2, 3))
        recall = self.rp_weight[0] * (true_positive + self.smooth) / (
            total_target + self.smooth)
        precision = self.rp_weight[1] * (true_positive + self.smooth) / (
            total_predict + self.smooth)
        class_wise_loss = 2 * recall * precision / (recall + precision)
        if class_weight is not None:
            class_wise_loss = class_wise_loss * class_weight
        loss = self.loss_weight * (1 - weight_reduce_loss(class_wise_loss,
            weight, reduction=reduction, avg_factor=avg_factor))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4, 1, 4, 4], dtype=torch.
        int64)]


def get_init_inputs():
    return [[], {}]

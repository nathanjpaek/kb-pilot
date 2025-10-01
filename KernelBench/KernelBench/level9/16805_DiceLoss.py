import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._C
import torch.serialization


def binary_dice_loss(pred, label, smooth=1e-05):
    """
    :param pred: [N, *]: here should be scores in [0,1]
    :param label: [N, *]
    :param power: 1 for abs, 2 for square
    :return: [N]
    """
    pred = pred.contiguous().view(pred.shape[0], -1).float()
    label = label.contiguous().view(label.shape[0], -1).float()
    num = 2 * torch.sum(torch.mul(pred, label), dim=1) + smooth
    den = torch.sum(pred, dim=1) + torch.sum(label, dim=1) + smooth
    loss = 1 - num / den
    return loss


def _make_one_hot(label, num_classes):
    """
    :param label: [N, *], values in [0,num_classes)
    :return: [N, C, *]
    """
    label = label.unsqueeze(1)
    shape = list(label.shape)
    shape[1] = num_classes
    result = torch.zeros(shape, device=label.device)
    result.scatter_(1, label, 1)
    return result


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


def dice_loss(pred_raw, label_raw, weight=None, class_weight=None,
    reduction='mean', avg_factor=None, ignore_class=-1, smooth=1e-05):
    """
    :param pred:  [N, C, *]scores without softmax
    :param label: [N, *]
    :return: reduction([N])
    """
    pred = pred_raw.clone()
    label = label_raw.clone()
    num_classes = pred.shape[1]
    if class_weight is not None:
        class_weight = class_weight.float()
    if pred.shape != label.shape:
        label = _make_one_hot(label, num_classes)
    pred = F.softmax(pred, dim=1)
    loss = 0.0
    for i in range(num_classes):
        if i != ignore_class:
            dice_loss = binary_dice_loss(pred[:, i], label[:, i], smooth)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            loss += dice_loss
    if ignore_class != -1:
        num_classes -= 1
    loss = loss / num_classes
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction,
        avg_factor=avg_factor)
    return loss


class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean',
        class_weight=None, loss_weight=1.0, ignore_class=-1, smooth=1e-05):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cls_criterion = dice_loss
        self.ignore_class = ignore_class
        self.smooth = smooth

    def forward(self, cls_score, label, weight=None, avg_factor=None,
        reduction_override=None, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.
            reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            assert class_weight.shape[0] == label.shape[1
                ], 'Expect weight shape [{}], get[{}]'.format(label.shape[1
                ], class_weight.shape[0])
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label,
            weight, class_weight=class_weight, reduction=reduction,
            avg_factor=avg_factor, ignore_class=self.ignore_class, smooth=
            self.smooth)
        return loss_cls


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

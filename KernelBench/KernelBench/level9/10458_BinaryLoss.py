import functools
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._C
import torch.serialization


def binary_ce_loss(pred, label, **kwargs):
    loss = F.binary_cross_entropy(pred, label, reduction='none')
    loss = torch.mean(loss, dim=(1, 2))
    return loss


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


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=
        None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth
    return 1 - num / den


def binary_ce_dice_loss(pred, label, smooth=1.0, **kwargs):
    loss1 = binary_ce_loss(pred, label, **kwargs)
    loss2 = binary_dice_loss(pred, label, smooth=smooth)
    return loss1 + loss2


def _make_one_hot(gt, num_classes, ignore=(0, 255)):
    """
    :param label: [N, *], values in [0,num_classes)
    :param ignore: ignore value of background, here is (0, 255)
    :return: [N, C, *]
    """
    label = gt
    label = label.unsqueeze(1)
    shape = list(label.shape)
    shape[1] = num_classes + 1
    if ignore is not None:
        if 0 in ignore:
            for index in ignore:
                label[label == index] = num_classes + 1
            label = label - 1
        else:
            for index in ignore:
                label[label == index] = num_classes
    result = torch.zeros(shape, device=label.device)
    result.scatter_(1, label, 1)
    return result[:, :-1]


def binary_loss(pred_raw, label_raw, loss_func, weight=None, class_weight=
    None, class_weight_norm=False, reduction='mean', avg_factor=None,
    smooth=1.0, **kwargs):
    """
    :param pred:  [N, C, *] scores without softmax
    :param label: [N, *] in [0, C], 0 stands for background, 1~C stands for pred in 0~C-1
    :return: reduction([N])
    """
    pred = pred_raw.clone()
    label = label_raw.clone()
    num_classes = pred.shape[1]
    if class_weight is not None:
        class_weight = class_weight.float()
    if pred.shape != label.shape:
        label = _make_one_hot(label, num_classes)
    pred = torch.sigmoid(pred)
    loss = 0.0
    for i in range(num_classes):
        if isinstance(loss_func, tuple):
            loss_function = loss_func[i]
        else:
            loss_function = loss_func
        class_loss = loss_function(pred[:, i], label[:, i], smooth=smooth)
        if class_weight is not None:
            class_loss *= class_weight[i]
        loss += class_loss
    if class_weight is not None and class_weight_norm:
        loss = loss / torch.sum(class_weight)
    else:
        loss = loss / num_classes
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction,
        avg_factor=avg_factor)
    return loss


class BinaryLoss(nn.Module):

    def __init__(self, loss_type='ce', reduction='mean', class_weight=None,
        class_weight_norm=False, loss_weight=1.0, smooth=1.0, **kwargs):
        super(BinaryLoss, self).__init__()
        assert loss_type in ['ce', 'dice', 'ce_dice']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.class_weight_norm = class_weight_norm
        self.loss_type = loss_type
        self.smooth = smooth

    def forward(self, cls_score, label, weight=None, avg_factor=None,
        reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.
            reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            assert class_weight.shape[0] == cls_score.shape[1
                ], 'Expect weight shape [{}], get[{}]'.format(cls_score.
                shape[1], class_weight.shape[0])
        else:
            class_weight = None
        loss_func = None
        if self.loss_type == 'ce':
            loss_func = binary_ce_loss
        elif self.loss_type == 'dice':
            loss_func = binary_dice_loss
        elif self.loss_type == 'ce_dice':
            loss_func = binary_ce_dice_loss
        loss_cls = self.loss_weight * binary_loss(cls_score, label,
            loss_func, weight, class_weight=class_weight, class_weight_norm
            =self.class_weight_norm, reduction=reduction, avg_factor=
            avg_factor, smooth=self.smooth)
        return loss_cls


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._C
import torch.serialization


def binary_cbce_loss(pred, label, **kwargs):
    """
    :param pred: [N, *]: here should be scores in [0,1]
    :param label: [N, *]: values in [0,1]
    :return: [N]
    """
    mask = (label > 0.5).float()
    b, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2]).float()
    num_neg = h * w - num_pos
    weight = torch.zeros_like(mask)
    pos_weight = num_neg / (num_pos + num_neg)
    neg_weight = num_pos / (num_pos + num_neg)
    for i in range(b):
        weight[i][label[i] > 0.5] = pos_weight[i]
        weight[i][label[i] <= 0.5] = neg_weight[i]
    loss = torch.nn.functional.binary_cross_entropy(pred.float(), label.
        float(), weight=weight, reduction='none')
    return loss


def binary_ce_loss(pred, label, **kwargs):
    loss = F.binary_cross_entropy(pred, label, reduction='none')
    loss = torch.mean(loss, dim=(1, 2))
    return loss


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


def binary_ce_dice_loss(pred, label, smooth=1.0, **kwargs):
    loss1 = binary_ce_loss(pred, label, **kwargs)
    loss2 = binary_dice_loss(pred, label, smooth=smooth)
    return loss1 + loss2


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
        assert loss_type in ['ce', 'dice', 'cbce', 'ce_dice', 'mix']
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
        elif self.loss_type == 'mix':
            loss_func = (binary_ce_loss, binary_ce_loss, binary_ce_loss,
                binary_dice_loss)
        elif self.loss_type == 'cbce':
            loss_func = binary_cbce_loss
        loss_cls = self.loss_weight * binary_loss(cls_score, label,
            loss_func, weight, class_weight=class_weight, class_weight_norm
            =self.class_weight_norm, reduction=reduction, avg_factor=
            avg_factor, smooth=self.smooth)
        return loss_cls


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

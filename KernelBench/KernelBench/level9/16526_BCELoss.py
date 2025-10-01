import torch
import torch.distributed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
import torch.utils.data
import torch.optim
import torch.optim.lr_scheduler


def bce_loss(pred, target, use_sigmoid=True):
    """Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    loss = func(pred_sigmoid, target, reduction='none')
    return loss.flatten()


class BCELoss(nn.Module):
    """
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
    """

    def __init__(self, use_sigmoid=True):
        super(BCELoss, self).__init__()
        self.use_sigmoid = use_sigmoid

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
        """
        return bce_loss(pred, target, use_sigmoid=self.use_sigmoid)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

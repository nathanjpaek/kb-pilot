import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """DiceLoss.

    .. seealso::
        Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for
        volumetric medical image segmentation." 2016 fourth international conference on 3D vision (3DV). IEEE, 2016.

    Args:
        smooth (float): Value to avoid division by zero when images and predictions are empty.

    Attributes:
        smooth (float): Value to avoid division by zero when images and predictions are empty.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()
        return -(2.0 * intersection + self.smooth) / (iflat.sum() + tflat.
            sum() + self.smooth)


class FocalLoss(nn.Module):
    """FocalLoss.

    .. seealso::
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        Proceedings of the IEEE international conference on computer vision. 2017.

    Args:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.

    Attributes:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.
    """

    def __init__(self, gamma=2, alpha=0.25, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1.0 - self.eps)
        cross_entropy = -(target * torch.log(input) + (1 - target) * torch.
            log(1 - input))
        logpt = -cross_entropy
        pt = torch.exp(logpt)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = -at * logpt
        focal_loss = balanced_cross_entropy * (1 - pt) ** self.gamma
        return focal_loss.sum()


class FocalDiceLoss(nn.Module):
    """FocalDiceLoss.

    .. seealso::
        Wong, Ken CL, et al. "3D segmentation with exponential logarithmic loss for highly unbalanced object sizes."
        International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.

    Args:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.

    Attributes:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
    """

    def __init__(self, beta=1, gamma=2, alpha=0.25):
        super().__init__()
        self.beta = beta
        self.focal = FocalLoss(gamma, alpha)
        self.dice = DiceLoss()

    def forward(self, input, target):
        dc_loss = -self.dice(input, target)
        fc_loss = self.focal(input, target)
        loss = torch.log(torch.clamp(fc_loss, 1e-07)) - self.beta * torch.log(
            torch.clamp(dc_loss, 1e-07))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

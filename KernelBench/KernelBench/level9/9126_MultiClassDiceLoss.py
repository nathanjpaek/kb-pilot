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


class MultiClassDiceLoss(nn.Module):
    """Multi-class Dice Loss.

    Inspired from https://arxiv.org/pdf/1802.10508.

    Args:
        classes_of_interest (list): List containing the index of a class which its dice will be added to the loss.
            If is None all classes are considered.

    Attributes:
        classes_of_interest (list): List containing the index of a class which its dice will be added to the loss.
            If is None all classes are considered.
        dice_loss (DiceLoss): Class computing the Dice loss.
    """

    def __init__(self, classes_of_interest=None):
        super(MultiClassDiceLoss, self).__init__()
        self.classes_of_interest = classes_of_interest
        self.dice_loss = DiceLoss()

    def forward(self, prediction, target):
        dice_per_class = 0
        n_classes = prediction.shape[1]
        if self.classes_of_interest is None:
            self.classes_of_interest = range(n_classes)
        for i in self.classes_of_interest:
            dice_per_class += self.dice_loss(prediction[:, i], target[:, i])
        return dice_per_class / len(self.classes_of_interest)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    """
    This class implements the lovasz hinge loss which is the continuous of the IoU for binary segmentation.
    Source: https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self) ->None:
        """
        Constructor method
        """
        super(LovaszHingeLoss, self).__init__()

    def _calc_grad(self, label_sorted: 'torch.Tensor') ->torch.Tensor:
        """
        Method computes the gradients of the sorted and flattened label
        :param label_sorted: (torch.Tensor) Sorted and flattened label of shape [n]
        :return: (torch.Tensor) Gradient tensor
        """
        label_sum = label_sorted.sum()
        intersection = label_sum - label_sorted.cumsum(dim=0)
        union = label_sum + (1 - label_sorted).cumsum(dim=0)
        iou = 1.0 - intersection / union
        iou[1:] = iou[1:] - iou[0:-1]
        return iou

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        signs = 2.0 * label - 1.0
        error = 1.0 - prediction * signs
        errors_sorted, permutation = torch.sort(error, dim=0, descending=True)
        label_sorted = label[permutation]
        grad = self._calc_grad(label_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


class DiceLoss(nn.Module):
    """
    This class implements the dice loss for multiple instances
    """

    def __init__(self, smooth_factor: 'float'=1.0) ->None:
        super(DiceLoss, self).__init__()
        self.smooth_factor = smooth_factor

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}, smooth factor={}'.format(self.__class__.__name__, self.
            smooth_factor)

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        loss = torch.tensor(1.0, dtype=torch.float32, device=prediction.device
            ) - (2.0 * torch.sum(torch.mul(prediction, label)) + self.
            smooth_factor) / (torch.sum(prediction) + torch.sum(label) +
            self.smooth_factor)
        return loss


class FocalLoss(nn.Module):
    """
    This class implements the segmentation focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: 'float'=0.25, gamma: 'float'=2.0) ->None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}, alpha={}, gamma={}'.format(self.__class__.__name__,
            self.alpha, self.gamma)

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        binary_cross_entropy_loss = -(label * torch.log(prediction.clamp(
            min=1e-12)) + (1.0 - label) * torch.log((1.0 - prediction).
            clamp(min=1e-12)))
        focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
        loss = ((1.0 - focal_factor) ** self.gamma *
            binary_cross_entropy_loss * self.alpha).sum(dim=1).mean()
        return loss


class SegmentationLoss(nn.Module):
    """
    This class implement the segmentation loss.
    """

    def __init__(self, dice_loss: 'nn.Module'=DiceLoss(), focal_loss:
        'nn.Module'=FocalLoss(), lovasz_hinge_loss: 'nn.Module'=
        LovaszHingeLoss(), w_dice: 'float'=1.0, w_focal: 'float'=0.2,
        w_lovasz_hinge: 'float'=0.0) ->None:
        super(SegmentationLoss, self).__init__()
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.lovasz_hinge_loss = lovasz_hinge_loss
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_lovasz_hinge = w_lovasz_hinge

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return ('{}, {}, w_focal={}, {}, w_dice={}, {}, w_lovasz_hinge={}'.
            format(self.__class__.__name__, self.dice_loss.__class__.
            __name__, self.w_dice, self.focal_loss.__class__.__name__, self
            .w_focal, self.lovasz_hinge_loss.__class__.__name__, self.
            w_lovasz_hinge))

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the segmentation loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Loss value
        """
        return self.w_dice * self.dice_loss(prediction, label
            ) + self.w_focal * self.focal_loss(prediction, label
            ) + self.w_lovasz_hinge * self.lovasz_hinge_loss(prediction, label)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

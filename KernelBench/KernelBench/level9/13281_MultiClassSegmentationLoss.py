import torch
import torch.nn as nn
from torch.autograd import Variable


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


class LovaszSoftmaxLoss(nn.Module):
    """
    Implementation of the Lovasz-Softmax loss.
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self) ->None:
        """
        Constructor method
        """
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        _, label = label.max(dim=0)
        classes, _height, _width = prediction.size()
        prediction = prediction.permute(1, 2, 0).contiguous().view(-1, classes)
        label = label.view(-1)
        losses = torch.zeros(classes, dtype=torch.float, device=prediction.
            device)
        for c in range(classes):
            fg = (label == c).float()
            class_prediction = prediction[:, c]
            errors = (Variable(fg) - class_prediction).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            p = len(fg_sorted)
            gts = fg_sorted.sum()
            intersection = gts - fg_sorted.float().cumsum(0)
            union = gts + (1 - fg_sorted).float().cumsum(0)
            jaccard = 1.0 - intersection / union
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            losses[c] = torch.dot(errors_sorted, Variable(jaccard))
        return losses.mean()


class FocalLossMultiClass(nn.Module):
    """
    Implementation of the multi class focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: 'float'=0.25, gamma: 'float'=2.0) ->None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        super(FocalLossMultiClass, self).__init__()
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
        cross_entropy_loss = -(label * torch.log(prediction.clamp(min=1e-12))
            ).sum(dim=0)
        focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
        loss = ((1.0 - focal_factor) ** self.gamma * cross_entropy_loss *
            self.alpha).sum(dim=0).mean()
        return loss


class MultiClassSegmentationLoss(nn.Module):
    """
    Multi class segmentation loss for the case if a softmax is utilized as the final activation.
    """

    def __init__(self, dice_loss: 'nn.Module'=DiceLoss(), focal_loss:
        'nn.Module'=FocalLossMultiClass(), lovasz_softmax_loss: 'nn.Module'
        =LovaszSoftmaxLoss(), w_dice: 'float'=1.0, w_focal: 'float'=0.1,
        w_lovasz_softmax: 'float'=0.0) ->None:
        super(MultiClassSegmentationLoss, self).__init__()
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.lovasz_softmax_loss = lovasz_softmax_loss
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_lovasz_softmax = w_lovasz_softmax

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return ('{}, {}, w_focal={}, {}, w_dice={}, {}, w_lovasz_softmax={}'
            .format(self.__class__.__name__, self.dice_loss.__class__.
            __name__, self.w_dice, self.focal_loss.__class__.__name__, self
            .w_focal, self.lovasz_softmax_loss.__class__.__name__, self.
            w_lovasz_softmax))

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
            ) + self.w_lovasz_softmax * self.lovasz_softmax_loss(prediction,
            label)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]

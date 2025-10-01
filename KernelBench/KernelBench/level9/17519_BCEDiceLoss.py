import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(smooth=1):
    """Create Dice Loss.

    Args:
        smooth (float, optional): Smoothing value. A larger
            smooth value (also known as Laplace smooth, or
            Additive smooth) can be used to avoid overfitting.
            (default: 1)
    
    Returns:
        Dice loss function
    """
    return DiceLoss(smooth=smooth)


def bce_loss():
    """Create Binary Cross Entropy Loss.
    The loss automatically applies the sigmoid activation
    function on the prediction input.

    Returns:
        Binary cross entropy loss function
    """
    return nn.BCEWithLogitsLoss()


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        """Dice Loss.

        Args:
            smooth (float, optional): Smoothing value. A larger
                smooth value (also known as Laplace smooth, or
                Additive smooth) can be used to avoid overfitting.
                (default: 1)
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        """Calculate Dice Loss.

        Args:
            input (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
        
        Returns:
            dice loss
        """
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()
        return 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class BCEDiceLoss(nn.Module):

    def __init__(self, smooth=1e-06):
        """BCEDice Loss.

        Args:
            smooth (float, optional): Smoothing value.
        """
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss(smooth)

    def forward(self, input, target):
        """Calculate BCEDice Loss.

        Args:
            input (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
        
        Returns:
            BCEDice loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(input, target)
        dice_loss = self.dice(torch.sigmoid(input), target)
        return bce_loss + 2 * dice_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

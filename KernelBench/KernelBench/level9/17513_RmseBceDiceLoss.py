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


def bce_dice_loss(smooth=1e-06):
    """Create BCEDice Loss.

    Args:
        smooth (float, optional): Smoothing value.
    
    Returns:
        BCEDice loss function
    """
    return BCEDiceLoss(smooth=smooth)


def rmse_loss(smooth=1e-06):
    """Create Root Mean Squared Error Loss.

    Returns:
        Root mean squared error loss function
    """
    return RMSELoss(smooth=1e-06)


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


class RMSELoss(nn.Module):

    def __init__(self, smooth=1e-06):
        """RMSE Loss.

        Args:
            smooth (float, optional): Smoothing value.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.smooth = smooth

    def forward(self, input, target):
        """Calculate RMSE Loss.

        Args:
            input (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
        
        Returns:
            RMSE loss
        """
        loss = torch.sqrt(self.mse(input, target) + self.smooth)
        return loss


class RmseBceDiceLoss(nn.Module):

    def __init__(self):
        super(RmseBceDiceLoss, self).__init__()
        self.rmse = rmse_loss()
        self.bce_dice = bce_dice_loss()

    def forward(self, prediction, label):
        return 2 * self.rmse(torch.sigmoid(prediction[0]), label[0]
            ) + self.bce_dice(prediction[1], label[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

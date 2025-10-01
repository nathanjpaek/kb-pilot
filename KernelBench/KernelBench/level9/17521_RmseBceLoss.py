import torch
import torch.nn as nn


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


class RmseBceLoss(nn.Module):

    def __init__(self):
        super(RmseBceLoss, self).__init__()
        self.rmse = rmse_loss()
        self.bce = bce_loss()

    def forward(self, prediction, label):
        return 2 * self.rmse(torch.sigmoid(prediction[0]), label[0]
            ) + self.bce(prediction[1], label[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

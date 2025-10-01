import torch
import torch.nn as nn


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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

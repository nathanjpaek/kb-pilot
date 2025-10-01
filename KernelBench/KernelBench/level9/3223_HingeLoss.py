import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    """Hinge loss function module for multi-label classification"""

    def __init__(self, margin=1.0, power=2, cost_weighted=False):
        """
        Args:
            margin (float, optional): margin for the hinge loss. Default 1.0
            power (int, optional): exponent for the hinge loss. Default to 2 for squared-hinge
            cost_weighted (bool, optional): whether to use label value as weight. Default False
        """
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.power = power
        self.cost_weighted = cost_weighted

    def forward(self, z, y, C_pos=1.0, C_neg=1.0):
        """Compute the hinge loss

        Args:
            z (torch.tensor): predicted matrix of size: (batch_size * output_size)
            y (torch.tensor): 0/1 ground truth of size: (batch_size * output_size)
            C_pos (float, optional): positive penalty for the hinge loss. Default 1.0
            C_neg (float, optional): negative penalty for the hinge loss. Default 1.0

        Returns:
            loss (torch.tensor): the tensor of average loss
        """
        y_binary = (y > 0).float()
        y_new = 2.0 * y_binary - 1.0
        loss = F.relu(self.margin - y_new * z)
        loss = loss ** self.power
        if self.cost_weighted:
            loss = loss * (C_pos * y + C_neg * (1.0 - y_binary))
        else:
            loss = loss * (C_pos * y_binary + C_neg * (1.0 - y_binary))
        return loss.mean(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.utils.data


class ModifiedSmoothL1Loss(nn.Module):

    def __init__(self, L1_regression_alpha: 'float'):
        super(ModifiedSmoothL1Loss, self).__init__()
        self.alpha = L1_regression_alpha

    def forward(self, normed_targets: 'torch.Tensor', pos_reg: 'torch.Tensor'):
        regression_diff = torch.abs(normed_targets - pos_reg)
        regression_loss = torch.where(torch.le(regression_diff, 1.0 / self.
            alpha), 0.5 * self.alpha * torch.pow(regression_diff, 2), 
            regression_diff - 0.5 / self.alpha)
        regression_loss = torch.where(torch.le(regression_diff, 0.01),
            torch.zeros_like(regression_loss), regression_loss)
        return regression_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'L1_regression_alpha': 4}]

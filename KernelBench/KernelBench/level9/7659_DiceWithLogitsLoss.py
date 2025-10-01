import torch
import torch.nn as nn
import torch.optim


class DiceWithLogitsLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super(DiceWithLogitsLoss, self).__init__()
        self.smooth = smooth

    def _dice_coeff(self, pred, target):
        """
        Args:
            pred: [N, 1] within [0, 1]
            target: [N, 1]
        Returns:
        """
        smooth = self.smooth
        inter = torch.sum(pred * target)
        z = pred.sum() + target.sum() + smooth
        return (2 * inter + smooth) / z

    def forward(self, pred, target):
        pred_score = torch.sigmoid(pred)
        return 1.0 - self._dice_coeff(pred_score, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

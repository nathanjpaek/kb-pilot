import torch
from torch import nn


class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.m = nn.Sigmoid()
        self.gamma = 1.0
        self.p = 2

    def forward(self, y_pred, y_true):
        pred_prob = self.m(y_pred)
        numerator = 2.0 * torch.sum(pred_prob * y_true, dim=1) + self.gamma
        denominator = torch.sum(pred_prob.pow(self.p) + y_true, dim=1
            ) + self.gamma
        dsc_i = 1.0 - numerator / denominator
        dice_loss = dsc_i.mean()
        return dice_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

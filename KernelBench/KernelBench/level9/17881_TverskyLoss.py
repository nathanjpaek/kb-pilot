import torch
from torch import nn


class TverskyLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self):
        super(TverskyLoss, self).__init__()
        self.m = nn.Sigmoid()
        self.gamma = 1.0
        self.p = 2
        self.alpha = 0.7

    def forward(self, y_pred, y_true):
        pred_prob = self.m(y_pred)
        true_pos = torch.sum(pred_prob * y_true, dim=1)
        numerator = true_pos + self.gamma
        denominator = torch.sum((1 - self.alpha) * pred_prob.pow(self.p) + 
            self.alpha * y_true, dim=1) + self.gamma
        tl_i = (1.0 - numerator / denominator).pow(0.75)
        tl_loss = tl_i.mean()
        return tl_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

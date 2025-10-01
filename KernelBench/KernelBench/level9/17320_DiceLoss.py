import torch
import torch.nn as nn


def f_score(pr, gt, beta=1, eps=1e-07, threshold=None, activation='sigmoid'):
    activation_fn = torch.nn.Sigmoid()
    pr = activation_fn(pr)
    if threshold is not None:
        pr = (pr > threshold).float()
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 
        2 * fn + fp + eps)
    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-07, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1.0, eps=self.eps, threshold=
            None, activation=self.activation)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

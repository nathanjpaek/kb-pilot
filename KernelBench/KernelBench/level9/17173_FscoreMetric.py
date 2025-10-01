import torch
import torch.nn as nn


def f_score(pr, gt, beta=1, eps=1e-07, threshold=0.5):
    """dice score(also referred to as F1-score)"""
    if threshold is not None:
        pr = (pr > threshold).float()
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 
        2 * fn + fp + eps)
    return score


class FscoreMetric(nn.Module):
    __name__ = 'f-score'

    def __init__(self, beta=1, eps=1e-07, threshold=0.5):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.beta = beta

    def forward(self, y_pr, y_gt):
        return f_score(y_pr, y_gt, self.beta, self.eps, self.threshold)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

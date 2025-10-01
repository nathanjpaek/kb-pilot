import torch
from torch import nn


def fb_loss(preds, trues, beta):
    smooth = 0.0001
    beta2 = beta * beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0.0, 1.0)
    TP = (preds * trues).sum(2)
    FP = (preds * (1 - trues)).sum(2)
    FN = ((1 - preds) * trues).sum(2)
    Fb = ((1 + beta2) * TP + smooth) / ((1 + beta2) * TP + beta2 * FN + FP +
        smooth)
    Fb = Fb * weights
    score = Fb.sum() / (weights.sum() + smooth)
    return torch.clamp(score, 0.0, 1.0)


class FBLoss(nn.Module):

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        return 1 - fb_loss(output, target, self.beta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes=5, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data, self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]

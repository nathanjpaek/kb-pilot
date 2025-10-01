import torch
from torch import nn


class weightedLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.thresholds = [0.5, 2, 5, 10, 30]
        self.weights = [1, 1, 2, 5, 10, 30]

    def forward(self, pred, label):
        weights = torch.ones_like(pred) * 3
        for i, threshold in enumerate(self.thresholds):
            weights = weights + (self.weights[i + 1] - self.weights[i]) * (
                label >= threshold).float()
        mse = torch.sum(weights * (pred - label) ** 2, (1, 3, 4))
        mae = torch.sum(weights * torch.abs(pred - label), (1, 3, 4))
        return (torch.mean(mse) + torch.mean(mae)) * 5e-06


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

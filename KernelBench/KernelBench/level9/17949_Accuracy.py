import torch
import torch.nn as nn


class Accuracy(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_true, y_pred):
        preds = (y_pred > self.threshold).int()
        return (preds == y_true).sum().float() / len(preds)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

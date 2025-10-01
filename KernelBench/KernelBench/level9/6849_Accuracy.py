import torch
import torch.nn as nn


class Accuracy(nn.Module):

    def __init__(self, binary=False):
        super().__init__()
        self.binary = binary

    def forward(self, preds, trues):
        if self.binary:
            preds = preds >= 0.5
        else:
            preds = preds.argmax(dim=1)
        result = (preds == trues).float().mean()
        return result

    def extra_repr(self):
        return f'binary={self.binary}'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

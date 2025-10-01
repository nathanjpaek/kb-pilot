import torch
import torch.nn as nn


def _assert_inputs(pred, true):
    assert pred.shape == true.shape, f'predition shape {pred.shape} is not the same as label shape {true.shape}'


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, binary=False):
        super().__init__()
        self.gamma = gamma
        self.binary = binary

    def forward(self, preds, trues):
        _assert_inputs(preds, trues)
        if self.binary:
            prob = trues * preds + (1 - trues) * (1 - preds)
        else:
            prob = (trues * preds).sum(dim=1)
        ln = (1 - prob).pow(self.gamma) * (prob + 1e-07).log()
        return -ln.mean()

    def extra_repr(self):
        return f'gamma={self.gamma}, binary={self.binary}'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


def _assert_inputs(pred, true):
    assert pred.shape == true.shape, f'predition shape {pred.shape} is not the same as label shape {true.shape}'


class FbetaLoss(nn.Module):

    def __init__(self, beta=1, axes=(0,), binary=False, smooth=1e-07):
        super().__init__()
        self.beta = beta
        self.axes = axes
        self.binary = binary
        self.smooth = smooth

    def forward(self, preds, trues):
        beta2 = self.beta ** 2
        if not self.binary:
            trues = trues[:, 1:, ...]
            preds = preds[:, 1:, ...]
        _assert_inputs(preds, trues)
        p = (beta2 + 1) * (trues * preds).sum(dim=self.axes)
        s = (beta2 * trues + preds).sum(dim=self.axes)
        fb = (p + self.smooth) / (s + self.smooth)
        return (1 - fb).mean()

    def extra_repr(self):
        return (
            f'beta={self.beta}, axes={self.axes}, binary={self.binary}, smooth={self.smooth}'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

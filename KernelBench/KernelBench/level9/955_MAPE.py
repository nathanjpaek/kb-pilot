import torch
import torch.nn as nn


class MAPE(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, actual):
        mape = 100 * self.l1(pred, actual) / torch.max(pred, actual)
        return mape.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

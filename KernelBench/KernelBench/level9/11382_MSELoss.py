import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """ Mean-squared error loss """

    def __init__(self, reduction='mean', eps=1e-08):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError(
                '`reduction` not recognized. must be "mean" or "sum"')
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        loss = (target - pred) ** 2
        loss = torch.mean(loss, 1)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / len(pred)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

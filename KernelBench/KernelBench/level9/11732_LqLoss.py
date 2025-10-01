import torch
from torch import nn


def lq_loss(y_pred, y_true, q):
    eps = 1e-07
    loss = y_pred * y_true
    loss = (1 - (loss + eps) ** q) / q
    return loss.mean()


class LqLoss(nn.Module):

    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, output, target):
        output = torch.sigmoid(output)
        return lq_loss(output, target, self.q)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

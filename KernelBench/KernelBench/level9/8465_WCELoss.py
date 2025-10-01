import torch
import torch.nn as nn
import torch.nn.functional as F


class WCELoss(nn.Module):

    def __init__(self):
        super(WCELoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        y_true = y_true / y_true.sum(2).sum(2, dtype=torch.float).unsqueeze(-1
            ).unsqueeze(-1)
        y_true[y_true != y_true] = 0.0
        y_true = torch.sum(y_true, dim=1, dtype=torch.float).unsqueeze(1)
        y_true = y_true * weights
        old_range = torch.max(y_true) - torch.min(y_true)
        new_range = 100 - 1
        y_true = (y_true - torch.min(y_true)) * new_range / old_range + 1
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred,
            dim=1)), dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred,
            dim=1)), dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

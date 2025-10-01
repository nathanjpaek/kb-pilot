import torch
import torch.nn as nn


class L_1st(nn.Module):

    def __init__(self, alpha):
        super(L_1st, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        Y = y_pred
        L = y_true
        batch_size = Y.shape[0]
        return 2 * self.alpha * torch.trace(torch.mm(torch.mm(Y.transpose(0,
            1), L), Y)) / batch_size


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]

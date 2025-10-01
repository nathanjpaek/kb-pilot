import torch
import torch.nn as nn


class BatchMeanCrossEntropyWithLogSoftmax(nn.Module):

    def forward(self, y_hat, y):
        return -(y_hat * y).sum(dim=1).mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

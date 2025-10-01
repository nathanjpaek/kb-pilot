import torch
import torch.nn as nn


class MyEntropy(nn.Module):

    def __init__(self):
        super(MyEntropy, self).__init__()

    def forward(self, predictions, target):
        b_size = predictions.size(0)
        lsm_func = nn.LogSoftmax(dim=1)
        logsoftmax = lsm_func(predictions)
        loss = -logsoftmax[torch.arange(b_size), target]
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]

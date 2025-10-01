import torch
import torch.nn as nn


class MyMetric(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        return pred.eq(target.view_as(pred)).sum() / output.size(0)


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {}]

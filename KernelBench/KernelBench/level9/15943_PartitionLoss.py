import torch
import torch.nn as nn


class PartitionLoss(nn.Module):

    def __init__(self):
        super(PartitionLoss, self).__init__()

    def forward(self, x):
        num_head = x.size(1)
        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / var)
        else:
            loss = 0
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

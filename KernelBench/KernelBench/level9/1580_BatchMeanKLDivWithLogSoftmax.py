import torch
import torch.nn as nn


class BatchMeanKLDivWithLogSoftmax(nn.Module):

    def forward(self, p, log_q, log_p):
        return (p * log_p - p * log_q).sum(dim=1).mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

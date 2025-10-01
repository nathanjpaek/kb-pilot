import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class GOODLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, ub_log_conf):
        return (ub_log_conf ** 2 / 2).log1p()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

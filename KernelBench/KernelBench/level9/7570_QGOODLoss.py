import math
import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class QGOODLoss(nn.Module):

    def __init__(self, quantile=0.8):
        super().__init__()
        self.quantile = quantile

    def forward(self, ub_log_conf):
        batch_size_out = ub_log_conf.shape[0]
        l = math.floor(batch_size_out * self.quantile)
        h = batch_size_out - l
        above_quantile_indices = ub_log_conf.topk(h, largest=True)[1]
        return (ub_log_conf[above_quantile_indices] ** 2 / 2).log1p()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

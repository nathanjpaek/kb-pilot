import torch
import torch.nn as nn


class CategoricalKLLoss(nn.Module):

    def __init__(self):
        super(CategoricalKLLoss, self).__init__()

    def forward(self, P, Q):
        log_P = P.log()
        log_Q = Q.log()
        kl = (P * (log_P - log_Q)).sum(dim=-1).sum(dim=-1)
        return kl.mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

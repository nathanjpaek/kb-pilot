import torch
import torch.nn as nn


class KLDiscCriterion(nn.Module):
    """
    calculate
    sum (j=1,...,K) D_KL[q(c_j|x)||p(c_j|x)]
    """

    def __init__(self):
        super(KLDiscCriterion, self).__init__()

    def forward(self, disc_log_pre, disc_gt, qp_order=True):
        batch_size = disc_log_pre.size(0)
        disc_log_gt = torch.log(disc_gt + 0.0001)
        if qp_order:
            loss = torch.sum(torch.exp(disc_log_pre) * (disc_log_pre -
                disc_log_gt)) / batch_size
        else:
            loss = torch.sum(disc_gt * (disc_log_gt - disc_log_pre)
                ) / batch_size
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

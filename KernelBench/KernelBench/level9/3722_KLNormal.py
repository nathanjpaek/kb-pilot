import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed


class KLNormal(nn.Module):

    def __init__(self):
        super(KLNormal, self).__init__()

    def forward(self, qm, qv, pm, pv):
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm -
            pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn.functional as F
import torch.nn as nn


class Entropy(nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        num, ms1, ms2 = x.size()
        ent_p2g = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        ent_g2p = F.softmax(x, dim=2) * F.log_softmax(x, dim=2)
        ent_sum = -1.0 * ent_p2g.view(num, -1).sum() - ent_g2p.view(num, -1
            ).sum()
        return ent_sum / (ms1 * ms2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]

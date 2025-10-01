import torch
import torch.nn as nn
import torch.nn.functional as F


class BCE_disc_sm_v6(nn.Module):

    def __init__(self, weight_list=None, lb_sm1=0.5, lb_sm0=0.1):
        super(BCE_disc_sm_v6, self).__init__()
        self.weight_list = weight_list
        self.lb_sm1 = lb_sm1
        self.lb_sm0 = lb_sm0

    def forward(self, x, labels):
        assert (x >= 0).all() and (x <= 1).all(), 'x is wrong'
        assert (labels >= 0).all() and (labels <= 1).all(), 'labels is wrong'
        labels = self.lb_sm1 * labels + self.lb_sm0 * (1 - labels)
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list,
            reduction='none')
        return loss.mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

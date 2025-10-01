import torch
import torch.nn as nn
import torch.nn.functional as F


class BCE_disc_sm_v5(nn.Module):

    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v5, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (labels >= 0).all() and (labels <= 1).all(), 'labels is wrong'
        labels = labels + self.lb_sm * (1 - labels)
        labels = labels / labels.sum(dim=1, keepdim=True)
        loss = -F.log_softmax(x, dim=1) * labels
        return loss.mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

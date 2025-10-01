import torch
import torch.nn.functional
import torch.nn as nn


class Attloss(nn.Module):

    def __init__(self):
        super(Attloss, self).__init__()
        self.maxvalueloss = 30

    def forward(self, x_org, att):
        d = torch.exp(6.0 * torch.abs(x_org - att))
        loss_att = (d - 1) / (d + 1)
        loss_att = loss_att.mean()
        loss_att = torch.clamp(loss_att, max=self.maxvalueloss)
        return 5.0 * loss_att


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

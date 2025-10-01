import torch
import torch.nn.functional as F


class Align(torch.nn.Module):

    def __init__(self, p):
        super(Align, self).__init__()
        self.p = p

    def forward(self, e1, e2):
        pred = -torch.norm(e1 - e2, p=self.p, dim=1)
        return pred

    def only_pos_loss(self, e1, r, e2):
        return -F.logsigmoid(-torch.sum(torch.pow(e1 + r - e2, 2), 1)).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 4}]

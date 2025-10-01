import torch
import torch.nn as nn


class OhemLoss(nn.Module):

    def __init__(self):
        super(OhemLoss, self).__init__()
        self.criteria = nn.BCELoss()

    def forward(self, label_p, label_t):
        label_p = label_p.view(-1)
        label_t = label_t.view(-1)
        loss = self.criteria(label_p, label_t)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

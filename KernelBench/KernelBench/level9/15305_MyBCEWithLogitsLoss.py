import torch
import torch.utils.data
import torch
import torch.nn as nn


class MyBCEWithLogitsLoss(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.m = nn.BCEWithLogitsLoss()

    def forward(self, positives, negatives):
        values = torch.cat((positives, negatives), dim=-1)
        labels = torch.cat((positives.new_ones(positives.size()), negatives
            .new_zeros(negatives.size())), dim=-1)
        return self.m(values, labels)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

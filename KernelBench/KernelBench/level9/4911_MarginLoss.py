import torch
import torch.nn.functional as F


class MarginLoss(torch.nn.Module):

    def __init__(self, margin, C=0, reverse=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.C = C
        if not isinstance(reverse, bool):
            raise TypeError('param reverse must be True or False!')
        self.reverse = 1 if reverse is False else -1

    def forward(self, positive_score, negative_score, penalty=0.0):
        output = torch.mean(F.relu(self.margin + self.reverse * (
            positive_score - negative_score))) + self.C * penalty
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]

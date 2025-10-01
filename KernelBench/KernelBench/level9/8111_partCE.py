import torch
import torch.nn as nn
import torch.utils.data


class partCE(nn.Module):

    def __init__(self, if_average=False):
        super(partCE, self).__init__()
        self.crit = nn.CrossEntropyLoss(size_average=if_average)
        self.maximum_score = 100000

    def forward(self, scores, target):
        par_scores = scores - (1 - target) * self.maximum_score
        _, max_ind = torch.max(par_scores, 1)
        max_ind = max_ind.detach()
        loss = self.crit(scores, max_ind)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

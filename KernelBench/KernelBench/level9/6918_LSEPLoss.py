import torch
import torch.nn as nn


def lsep_loss_stable(input, target, average=True):
    n = input.size(0)
    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()
    differences = differences.view(n, -1)
    where_lower = where_lower.view(n, -1)
    max_difference, _index = torch.max(differences, dim=1, keepdim=True)
    differences = differences - max_difference
    exps = differences.exp() * where_lower
    lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1)
        )
    if average:
        return lsep.mean()
    else:
        return lsep


class LSEPLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        return lsep_loss_stable(preds, targets)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

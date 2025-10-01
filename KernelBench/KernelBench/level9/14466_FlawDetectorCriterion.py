import torch
import torch.nn as nn
import torch.nn.functional as F


class FlawDetectorCriterion(nn.Module):
    """ Criterion of the flaw detector.
    """

    def __init__(self):
        super(FlawDetectorCriterion, self).__init__()

    def forward(self, pred, gt, is_ssl=False, reduction=True):
        loss = F.mse_loss(pred, gt, reduction='none')
        if reduction:
            loss = torch.mean(loss, dim=(1, 2, 3))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
from torch import nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):

    def __init__(self, weights=None):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit, truth):
        batch_size, num_class = truth.shape
        logit = logit.view(batch_size, num_class)
        truth = truth.view(batch_size, num_class)
        assert logit.shape == truth.shape
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction=
            'none')
        if self.weights is None:
            loss = loss.mean()
        else:
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()
            pos_sum = pos.sum().item() + 1e-12
            neg_sum = neg.sum().item() + 1e-12
            loss = (self.weights[1] * pos * loss / pos_sum + self.weights[0
                ] * neg * loss / neg_sum).sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]

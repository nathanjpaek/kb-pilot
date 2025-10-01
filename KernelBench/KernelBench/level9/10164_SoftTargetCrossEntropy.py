import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class SoftTargetCrossEntropy(nn.Module):
    """
    The native CE loss with soft target
    input: x is output of model, target is ground truth
    return: loss
    """

    def __init__(self, weights):
        super(SoftTargetCrossEntropy, self).__init__()
        self.weights = weights

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1) * self.weights,
            dim=-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weights': 4}]

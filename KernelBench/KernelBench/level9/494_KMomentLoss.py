import torch
import torch.optim
import torch.nn as nn


class KMomentLoss(nn.Module):

    def __init__(self, k: 'int'=4):
        """
        k moment distance, where `k` represents the highest order of moment.
        """
        super(KMomentLoss, self).__init__()
        self.eps = 1e-08
        self.k = k

    def euclidean_dist(self, d1: 'torch.Tensor', d2: 'torch.Tensor'
        ) ->torch.Tensor:
        return (((d1 - d2) ** 2).sum() + self.eps).sqrt()

    def forward(self, f1: 'torch.Tensor', f2: 'torch.Tensor') ->torch.Tensor:
        loss = 0.0
        for order in range(1, self.k + 1):
            f1_k = (f1 ** order).mean(dim=0)
            f2_k = (f2 ** order).mean(dim=0)
            loss += self.euclidean_dist(f1_k, f2_k)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

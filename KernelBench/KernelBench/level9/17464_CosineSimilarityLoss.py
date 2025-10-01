from torch.nn import Module
import torch
import torch.nn.functional as F


class BaseLoss(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CosineSimilarityLoss(BaseLoss):

    def __init__(self, dim=1, eps=1e-08, reduction='mean', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        sim = 1 - F.cosine_similarity(output, target, self.dim, self.eps)
        if self.reduction == 'mean':
            return sim.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

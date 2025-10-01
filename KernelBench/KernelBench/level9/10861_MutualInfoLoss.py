import torch
from torch import nn


class MutualInfoLoss(nn.Module):
    """
        Mutual Information Loss derived from ss-with-RIM that also applied in
        this work.
        First term enforces to generate a sparse nSpixel dimension vector for
        each pixel; Second term indicates the cardinality of each spixel.

        Args:
            logits: torch.tensor
                A trainable tensor of shape (b, nSpixel, h, w) that
                represents the probability of each pixel belonging to all spixels.
                It should be softmaxed before calling this loss funtion.

            coef: float
                A coefficient that controls the amplitude of second term.
    """

    def __init__(self, coef=2):
        super().__init__()
        self.coef = coef

    def forward(self, logits):
        pixel_wise_ent = -(logits * torch.log(logits + 1e-16)).sum(1).mean()
        marginal_prob = logits.mean((2, 3))
        marginal_ent = -(marginal_prob * torch.log(marginal_prob + 1e-16)).sum(
            1).mean()
        return pixel_wise_ent - self.coef * marginal_ent


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

import torch
import torch.nn as nn


class L1CosineSim(nn.Module):
    """ L1 loss with Cosine similarity.
    Can be used to replace L1 pixel loss, but includes a cosine similarity term
    to ensure color correctness of the RGB vectors of each pixel.
    lambda is a constant factor that adjusts the contribution of the cosine similarity term
    It provides improved color stability, especially for low luminance values, which
    are frequent in HDR images, since slight variations in any of the RGB components of these
    low values do not contribute much totheL1loss, but they may however cause noticeable
    color shifts.
    Ref: https://arxiv.org/pdf/1803.02266.pdf
    https://github.com/dmarnerides/hdr-expandnet/blob/master/train.py
    """

    def __init__(self, loss_lambda=5, reduction='mean'):
        super(L1CosineSim, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.loss_lambda = loss_lambda

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

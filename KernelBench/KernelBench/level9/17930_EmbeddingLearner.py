import torch
from torch import nn


class EmbeddingLearner(nn.Module):

    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, r, t):
        if r.dim() == 1:
            r = r.unsqueeze(0)
        h = h.view(1, -1, h.shape[-1])
        t = t.view(1, -1, t.shape[-1])
        r = r.view(r.shape[0], -1, r.shape[-1])
        score = h * r * t
        score = torch.sum(score, -1)
        return -score


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]

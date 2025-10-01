import torch
import torch.nn as nn
import torch.nn.functional as F


class AttCosine(torch.nn.Module):
    """
    AttCosine: Cosine attention that can be used by the Alignment module.
    """

    def __init__(self, softmax=True):
        super().__init__()
        self.softmax = softmax
        self.pdist = nn.CosineSimilarity(dim=3)

    def forward(self, query, y):
        att = self.pdist(query.unsqueeze(2), y.unsqueeze(1))
        sim = att.max(2)[0].unsqueeze(1)
        if self.softmax:
            att = F.softmax(att, dim=2)
        return att, sim


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]

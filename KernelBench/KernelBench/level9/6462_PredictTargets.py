import torch
from torch import nn
from torch.nn import functional as F


class PredictTargets(nn.Module):

    def __init__(self, dim):
        super(PredictTargets, self).__init__()
        self.linear1 = nn.Linear(2 * dim, dim)
        self.linear2 = nn.Linear(dim, 1)

    def forward(self, targets, embeddings):
        nt, b, vs = targets.shape
        ne = embeddings.size(0)
        vectors = torch.cat((targets.unsqueeze(1).expand(nt, ne, b, vs),
            embeddings.unsqueeze(0).expand(nt, ne, b, vs)), 3)
        return self.linear2(F.tanh(self.linear1(vectors))).squeeze(3)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]

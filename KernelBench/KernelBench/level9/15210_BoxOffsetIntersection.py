import torch
import torch.nn as nn
import torch.nn.functional as F


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layers = nn.Parameter(torch.zeros(self.dim * 2 + 2, self.dim))
        nn.init.xavier_uniform_(self.layers[:self.dim * 2, :])

    def forward(self, embeddings):
        w1, w2, b1, b2 = torch.split(self.layers, [self.dim, self.dim, 1, 1
            ], dim=0)
        layer1_act = F.relu(F.linear(embeddings, w1, b1.view(-1)))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(F.linear(layer1_mean, w2, b2.view(-1)))
        offset, _ = torch.min(embeddings, dim=0)
        return offset * gate


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]

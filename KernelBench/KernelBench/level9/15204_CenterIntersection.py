import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layers = nn.Parameter(torch.zeros(self.dim * 2 + 2, self.dim))
        nn.init.xavier_uniform_(self.layers[:self.dim * 2, :])

    def forward(self, embeddings):
        w1, w2, b1, b2 = torch.split(self.layers, [self.dim, self.dim, 1, 1
            ], dim=0)
        layer1_act = F.relu(F.linear(embeddings, w1, b1.view(-1)))
        attention = F.softmax(F.linear(layer1_act, w2, b2.view(-1)), dim=0)
        embedding = torch.sum(attention * embeddings, dim=0)
        return embedding


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]

import torch
from torch import nn
import torch.nn.functional as F


class NLKDifferenceCenter(nn.Module):

    def __init__(self, dim, hidden_dim):
        super(NLKDifferenceCenter, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(self.dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        attention = F.softmax(self.layer2(F.relu(self.layer1(emb))), dim=0)
        return torch.sum(attention * emb, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidden_dim': 4}]

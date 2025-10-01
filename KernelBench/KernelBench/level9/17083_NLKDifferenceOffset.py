import torch
from torch import nn
import torch.nn.functional as F


class NLKDifferenceOffset(nn.Module):

    def __init__(self, dim, hidden_dim):
        super(NLKDifferenceOffset, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(self.dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, loffset, z):
        all_emb = torch.cat((loffset.unsqueeze(0), z), dim=0)
        attention = F.softmax(self.layer2(F.relu(self.layer1(all_emb))), dim=0)
        return attention


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidden_dim': 4}]

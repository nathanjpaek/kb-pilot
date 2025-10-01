import torch
import torch.nn as nn
from scipy.stats import entropy as entropy
from scipy.spatial.distance import cosine as cosine


class Gate(nn.Module):

    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.transform = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.kaiming_normal_(self.transform.weight)

    def forward(self, query, key):
        r = self.transform(torch.cat((query.expand(key.size(0), -1), key), -1))
        gate = torch.sigmoid(r)
        return gate


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]

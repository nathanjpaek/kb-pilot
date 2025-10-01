import torch
import torch.nn as nn
import torch.nn.functional as F


class FairDiscriminator(nn.Module):

    def __init__(self, nfeat, nhid, nclass):
        """Just a simple MLP"""
        super(FairDiscriminator, self).__init__()
        self.hidden_layer = nn.Linear(nfeat, nhid)
        self.output_layer = nn.Linear(nhid, nclass)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        x = F.relu(self.output_layer(x))
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4}]

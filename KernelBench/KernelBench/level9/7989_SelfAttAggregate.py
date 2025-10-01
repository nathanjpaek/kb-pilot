import math
import torch
import torch.nn as nn


class SelfAttAggregate(torch.nn.Module):

    def __init__(self, agg_dim):
        super(SelfAttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(agg_dim, 1))
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hiddens, avgsum='sum'):
        """
        hiddens: (10, 19, 256, 100)
        """
        maxpool = torch.max(hiddens, dim=1)[0]
        if avgsum == 'sum':
            avgpool = torch.sum(hiddens, dim=1)
        else:
            avgpool = torch.mean(hiddens, dim=1)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1)
        energy = torch.bmm(agg_spatial.permute([0, 2, 1]), agg_spatial)
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute([0, 2, 1]))
        weight = self.weight.unsqueeze(0).repeat([hiddens.size(0), 1, 1])
        agg_feature = torch.bmm(weighted_feat.permute([0, 2, 1]), weight)
        return agg_feature.squeeze(dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'agg_dim': 4}]

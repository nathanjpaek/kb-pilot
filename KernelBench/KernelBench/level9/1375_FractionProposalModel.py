import torch
from torch import nn
from torch.nn import functional as F


class FractionProposalModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(FractionProposalModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        self.layer.bias.data.zero_()

    def forward(self, x):
        x = self.layer(x)
        x = F.softmax(x, dim=-1)
        ent = -torch.sum(x * torch.log(x), dim=-1)
        taus = torch.cumsum(x, -1)
        tau_0 = torch.zeros((x.size(0), 1), device=taus.device)
        taus = torch.concat([tau_0, taus], -1)
        tau_hats = (taus[:, :-1] + taus[:, 1:]) / 2
        return taus, tau_hats, ent


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]

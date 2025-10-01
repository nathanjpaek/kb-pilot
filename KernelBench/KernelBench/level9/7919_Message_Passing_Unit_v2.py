import torch
import torch.nn as nn
import torch.nn.functional as F


class Message_Passing_Unit_v2(nn.Module):

    def __init__(self, fea_size, filter_size=128):
        super(Message_Passing_Unit_v2, self).__init__()
        self.w = nn.Linear(fea_size, filter_size, bias=True)
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.
                size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.
                size()[1])
        gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
        gate = F.sigmoid(gate.sum(1))
        output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'fea_size': 4}]

import torch
from torchvision.transforms import functional as F
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class Message_Passing_Unit_v1(nn.Module):

    def __init__(self, fea_size, filter_size=128):
        super(Message_Passing_Unit_v1, self).__init__()
        self.w = nn.Linear(fea_size * 2, filter_size, bias=True)
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.
                size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.
                size()[1])
        gate = torch.cat([unary_term, pair_term], 1)
        gate = F.relu(gate)
        gate = torch.sigmoid(self.w(gate)).mean(1)
        output = pair_term * gate.view(-1, 1).expand(gate.size()[0],
            pair_term.size()[1])
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'fea_size': 4}]

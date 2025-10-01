import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        if requires_grad:
            self.beta = Parameter(torch.Tensor(1).uniform_(0, 1),
                requires_grad=requires_grad)
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj):
        norm2 = torch.norm(x, 2, 1).view(-1, 1)
        cos = self.beta * torch.div(torch.mm(x, x.t()), torch.mm(norm2,
            norm2.t()) + 1e-07)
        mask = (1.0 - adj) * -1000000000.0
        masked = cos + mask
        P = F.softmax(masked, dim=1)
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]

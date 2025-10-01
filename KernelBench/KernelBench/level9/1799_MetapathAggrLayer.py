import torch
from torch.nn import functional as F
from torch import nn


class MetapathAggrLayer(nn.Module):
    """
    metapath attention layer.
    """

    def __init__(self, in_features, nmeta, dropout, alpha):
        super(MetapathAggrLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.alpha = alpha
        self.n_meta = nmeta
        self.a = nn.Parameter(torch.zeros(size=(in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        input = input.transpose(0, 1)
        N = input.size()[0]
        a_input = input
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        e = F.softmax(e, dim=1)
        output = [torch.matmul(e[i], input[i]).unsqueeze(0) for i in range(N)]
        output = torch.cat(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'nmeta': 4, 'dropout': 0.5, 'alpha': 4}]

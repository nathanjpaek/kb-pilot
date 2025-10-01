from torch.nn import Module
import math
import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class Chebyshev_GL(Module):
    """
    GCN k-hop Layers
    x' = Sigma^k-1 (Z^k * w0^k), Z^k= polynomial
    """

    def __init__(self, in_features, out_features, k_hop, bias=True):
        super(Chebyshev_GL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_hop = k_hop
        self.weight = Parameter(torch.FloatTensor(k_hop, in_features,
            out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, laplacian):
        tx_0 = input
        out = torch.matmul(input, self.weight[0])
        if self.weight.size(0) > 1:
            tx_1 = torch.matmul(laplacian, input)
            out = out + torch.matmul(tx_1, self.weight[1])
        for k in range(2, self.weight.size(0)):
            tx_2 = 2 * torch.matmul(laplacian, tx_1) - tx_0
            out = out + torch.matmul(tx_2, self.weight[k])
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + 'with K={' + str(self.
            weight.size(0)) + '})'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'k_hop': 4}]

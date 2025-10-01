import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Parameter
import torch.nn
import torch.autograd


class SimpleGCN(nn.Module):
    """A simple graph convolution layer, similar to the one defined in
    Kipf et al. https://arxiv.org/abs/1609.02907

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::

            @article{kipf2016semi,
              title={Semi-Supervised Classification with Graph Convolutional Networks},
              author={Kipf, Thomas N and Welling, Max},
              journal={arXiv preprint arXiv:1609.02907},
              year={2016}
            }

    """

    def __init__(self, in_features, out_features, bias=True):
        super(SimpleGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 6.0 / math.sqrt(self.weight1.size(1) + self.weight1.size(0))
        stdv *= 0.6
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight1)
        side_len = max(support.shape[1] // 3, 2)
        if adj.type() == 'torch.cuda.sparse.FloatTensor':
            norm = torch.sparse.mm(adj, torch.ones((support.shape[0], 1)))
            normalized_support = support[:, :side_len] / norm
            side_1 = torch.sparse.mm(adj, normalized_support)
        else:
            side_1 = torch.mm(adj, support[:, :side_len])
        side_2 = support[:, side_len:]
        output = torch.cat((side_1, side_2), dim=1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]

import torch
import torch.nn as nn
from torch.nn import init


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)
        mul_L = self.cheb_polynomial(L).unsqueeze(1)
        result = torch.matmul(mul_L, inputs)
        result = torch.matmul(result, self.weight)
        result = torch.sum(result, dim=0) + self.bias
        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=
            laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device,
            dtype=torch.float)
        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian,
                        multi_order_laplacian[k - 1]) - multi_order_laplacian[
                        k - 2]
        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype
                ) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_c': 4, 'out_c': 4, 'K': 4}]

import torch
from torch import nn
from torch.nn import init


class GraphConvolution(nn.Module):

    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(window_size, in_features,
            out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        """
        :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
        :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
        :return output: FloatTensor (batch_size, window_size, node_num, out_features)
        """
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size,
            in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'window_size': 4, 'in_features': 4, 'out_features': 4}]

from torch.nn import Module
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphLearner(Module):

    def __init__(self, in_feature_dim, combined_feature_dim, n_obj, dropout=0.0
        ):
        """
        eq(1): A=EE^T, build adj matrix
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - combined_feature_dim: dimensionality of the joint hidden embedding
        - k: number of graph nodes/objects on the image
        """
        super(GraphLearner, self).__init__()
        self.in_dim = in_feature_dim
        self.combined_dim = combined_feature_dim
        self.n_obj = n_obj
        self.edge_layer_1 = nn.Linear(in_feature_dim, combined_feature_dim)
        self.edge_layer_2 = nn.Linear(combined_feature_dim,
            combined_feature_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, graph_nodes):
        """
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        """
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)
        h = self.edge_layer_2(h)
        h = F.relu(h)
        h = h.view(-1, self.n_obj, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        return adjacency_matrix


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature_dim': 4, 'combined_feature_dim': 4, 'n_obj': 4}]

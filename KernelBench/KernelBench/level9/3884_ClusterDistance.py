import torch
from torch import nn
from typing import Optional


class ClusterDistance(nn.Module):

    def __init__(self, n_classes: 'int', enc_shape: 'int', cluster_centers:
        'Optional[torch.Tensor]'=None) ->None:
        """

        :param n_classes: number of clusters
        :param enc_shape: embedding dimension of feature vectors
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super().__init__()
        self.enc_shape = enc_shape
        self.n_classes = n_classes
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_classes, self.
                enc_shape, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """

        :param x: FloatTensor of [batch size, embedding dimension]
        :param y: FloatTensor of [batch size,]
        :return: FloatTensor [batch size, number of clusters]
        """
        return torch.cdist(x, self.cluster_centers)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_classes': 4, 'enc_shape': 4}]

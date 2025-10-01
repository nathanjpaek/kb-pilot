import torch
from torch import nn
from typing import Optional


class ClusterAssignment(nn.Module):

    def __init__(self, n_classes: 'int', enc_shape: 'int', alpha: 'float'=
        1.0, cluster_centers: 'Optional[torch.Tensor]'=None) ->None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used to measure similarity between feature vector and each
        cluster centroid.
        :param n_classes: number of clusters
        :param enc_shape: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super().__init__()
        self.enc_shape = enc_shape
        self.n_classes = n_classes
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_classes, self.
                enc_shape, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, batch: 'torch.Tensor') ->torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers
            ) ** 2, 2)
        numerator = 1.0 / (1.0 + norm_squared / self.alpha)
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_classes': 4, 'enc_shape': 4}]

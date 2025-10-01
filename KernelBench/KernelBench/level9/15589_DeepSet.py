import torch
import torch.nn as nn


class DeepSet(nn.Module):
    """Aggregate object-level embeddings with a mean reduction.

    This module evaluates each object individually (using a object level
    embedding) and then aggregates the embeddings with a mean reduction.

    Parameters
    ----------
    n_features : int
        The number of features per object.

    embedding_size : int
        The target embedding size.

    embedding_module : torch module
        An uninitialized torch module that expects two parameters: the input
        and the output size. It should then act similar to ``nn.Linear``, i.e.
        transform only the last dimension of the input. Defaults to a simple
        linear module.
    """

    def __init__(self, n_features: 'int', embedding_size: 'int',
        embedding_module: 'nn.Module'=nn.Linear):
        super().__init__()
        self.embedding_module = embedding_module(n_features, embedding_size)

    def forward(self, instances):
        """Forward inputs through the network.

        Parameters
        ----------
        instances : tensor
            The input tensor of shape (N, *, O, F), where F is the number of
            features and O is the number of objects.

        Returns
        -------
        tensor
            A tensor of shape (N, *, E), where E ist the embedding size.
        """
        embedded_objects = self.embedding_module(instances)
        return torch.mean(embedded_objects, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'embedding_size': 4}]

import torch
import torch.utils.tensorboard
import torch.utils.data
import torch.distributed


class NumericalFeaturesEmbedding(torch.nn.Module):
    """Transform a sequence of numerical feature vectors into a single vector.

    Currently, this module simply aggregates the features by averaging, although more
    elaborate aggregation schemes (e.g. RNN) could be chosen.
    """

    def __init__(self, embedding_dim):
        super(NumericalFeaturesEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, embeddings):
        return embeddings.mean(axis=-2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4}]
